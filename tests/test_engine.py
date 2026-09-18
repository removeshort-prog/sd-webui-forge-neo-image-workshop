import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageOps, PngImagePlugin

from forge_image_workshop.batch import collect_sources, run_batch
from forge_image_workshop.engine import (
    Cancelled, LoadedImage, Options, apply_watermarks, load_image, process_image, save_image, upscale_rgba,
)
from forge_image_workshop.forge_adapter import upscale


def sample():
    image = Image.new("RGBA", (13, 9), (220, 60, 30, 0))
    image.putalpha(Image.frombytes("L", image.size, bytes((i * 17) % 256 for i in range(117))))
    return image


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_comfy_inverted_mask_and_red_channel(self):
        calls = []

        def model(image, *args):
            calls.append(image.copy())
            # Deliberately asymmetric transform distinguishes alpha from 1-alpha.
            channels = image.resize((26, 18), Image.Resampling.NEAREST).split()
            red = channels[0].point(lambda value: value // 2)
            return Image.merge("RGB", (red, Image.new("L", red.size, 0), Image.new("L", red.size, 255)))

        source = sample()
        output = upscale_rgba(source, Options(scale=2, model="test"), model)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].getchannel("R").tobytes(), ImageOps.invert(source.getchannel("A")).tobytes())
        expected = ImageOps.invert(source.getchannel("A")).resize((26, 18), Image.Resampling.NEAREST).point(lambda value: 255 - value // 2)
        self.assertEqual(output.getchannel("A").tobytes(), expected.tobytes())
        self.assertEqual(output.size, (26, 18))

    def test_opaque_image_skips_mask_model(self):
        calls = []

        def model(image, *args):
            calls.append(image)
            return image.resize((26, 18))

        output = upscale_rgba(Image.new("RGBA", (13, 9), (10, 20, 30, 255)), Options(scale=2, model="AI"), model)
        self.assertEqual(len(calls), 1)
        self.assertEqual(output.getchannel("A").getextrema(), (255, 255))

    def test_fast_alpha_and_downscale(self):
        calls = []

        def model(image, *args):
            calls.append(image)
            return image.resize((26, 18))

        source = sample()
        result = upscale_rgba(source, Options(scale=2, model="AI", alpha_mode="Lanczos（快速）"), model)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.getchannel("A").tobytes(), source.getchannel("A").resize((26, 18), Image.Resampling.LANCZOS).tobytes())
        reduced = upscale_rgba(source, Options(scale=0.65, model="AI"), lambda *args: self.fail("downscale ran AI"))
        self.assertEqual(reduced.size, (8, 6))

    def test_template_matching_and_alpha_not_multiplied_twice(self):
        templates = [("portrait", Image.new("RGBA", (10, 20), (0, 0, 255, 128))),
                     ("landscape", Image.new("RGBA", (20, 10), (255, 0, 0, 128)))]
        base = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        result, name = apply_watermarks(base, Options(image_enabled=True, preserve_alpha=False), templates)
        self.assertEqual(name, "landscape")
        self.assertEqual(result.getpixel((4, 4)), (255, 0, 0, 128))
        base = sample()
        result, _ = apply_watermarks(base, Options(image_enabled=True, preserve_alpha=True), templates)
        self.assertEqual(result.getchannel("A").tobytes(), base.getchannel("A").tobytes())

    def test_text_tiles_preserve_alpha(self):
        base = Image.new("RGBA", (350, 210), (10, 20, 30, 128))
        result, _ = apply_watermarks(base, Options(text_enabled=True, text="Watermark", text_tile=True, text_size=10))
        self.assertEqual(result.getchannel("A").tobytes(), base.getchannel("A").tobytes())
        self.assertNotEqual(result.convert("RGB").tobytes(), base.convert("RGB").tobytes())

    def test_large_logo_respects_margins_without_clipping(self):
        base = Image.new("RGBA", (100, 200))
        templates = [("logo", Image.new("RGBA", (200, 20), (255, 0, 0, 255)))]
        options = Options(image_enabled=True, image_mode="等比 Logo", image_width=100,
                          image_position="右下", margin=10, preserve_alpha=False)
        result, _ = apply_watermarks(base, options, templates)
        self.assertEqual(result.getchannel("A").getbbox(), (10, 182, 90, 190))

    def test_webp_png_tiff_preserve_alpha_and_pixels(self):
        source = sample()
        for fmt in ("WEBP", "PNG", "TIFF"):
            with self.subTest(fmt=fmt):
                path = self.root / (fmt + ".image")
                save_image(source, path, fmt, Options(webp_lossless=True))
                with Image.open(path) as read:
                    self.assertEqual(read.convert("RGBA").tobytes(), source.tobytes())

    def test_jpeg_matte(self):
        source = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
        path = self.root / "matte.jpg"
        save_image(source, path, "JPEG", Options(background="#00ff00", quality=100))
        with Image.open(path) as read:
            r, g, b = read.getpixel((10, 10))
            self.assertLess(r, 3)
            self.assertGreater(g, 252)
            self.assertLess(b, 3)
            self.assertEqual(read.mode, "RGB")

    def test_metadata_clear_and_preserve(self):
        source_path = self.root / "source.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "private prompt")
        sample().save(source_path, pnginfo=info)
        loaded = load_image(source_path)
        for clear in (True, False):
            path = self.root / f"output-{clear}.png"
            save_image(loaded.image, path, "PNG", Options(strip_metadata=clear), loaded.metadata)
            with Image.open(path) as read:
                self.assertEqual("parameters" in read.info, not clear)

    def test_exif_orientation_and_palette_transparency(self):
        source = Image.new("RGB", (20, 10), "red")
        exif = source.getexif()
        exif[274] = 6
        path = self.root / "rotate.jpg"
        source.save(path, exif=exif)
        loaded = load_image(path)
        self.assertEqual(loaded.image.size, (10, 20))
        self.assertNotIn(274, loaded.image.getexif())
        palette = Image.new("P", (10, 10))
        palette.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
        palette.putpixel((5, 5), 1)
        path = self.root / "palette.png"
        palette.save(path, transparency=0)
        rgba = load_image(path).image
        self.assertEqual(rgba.getpixel((0, 0))[3], 0)
        self.assertEqual(rgba.getpixel((5, 5))[3], 255)

    def test_animated_input_rejected(self):
        path = self.root / "animation.webp"
        Image.new("RGB", (20, 20), "red").save(path, save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")], duration=100)
        with self.assertRaisesRegex(ValueError, "动画"):
            load_image(path)

    def test_pixel_limit_and_nonfinite_scale(self):
        with self.assertRaises(ValueError):
            process_image(LoadedImage(Image.new("RGBA", (1000, 1000)), {}), Options(scale=8, max_megapixels=1))
        with self.assertRaises(ValueError):
            Options(scale=float("nan")).validate()

    def test_exclusive_save(self):
        path = self.root / "protected.png"
        path.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            save_image(sample(), path, "PNG", Options())
        self.assertEqual(path.read_bytes(), b"original")

    def test_model_silent_failure_is_reported(self):
        scaler = SimpleNamespace(scale=1, do_upscale=lambda image, path: image)
        shared = SimpleNamespace(sd_upscalers=[SimpleNamespace(name="bad", scaler=scaler, data_path="test", scale=2)])
        with patch.dict("sys.modules", {"modules": SimpleNamespace(shared=shared)}):
            with self.assertRaisesRegex(RuntimeError, "没有放大"):
                upscale(sample().convert("RGB"), "bad", 2, 64, lambda: None)

    def test_batch_unique_names_report_zip_and_error_recovery(self):
        for folder in ("a", "b"):
            (self.root / folder).mkdir()
            sample().save(self.root / folder / "same.png")
        corrupt = self.root / "bad.png"
        corrupt.write_bytes(b"not an image")
        sources = [self.root / "a/same.png", corrupt, self.root / "b/same.png"]
        result = run_batch(sources, self.root / "out", Options(formats=("PNG", "WEBP")))
        self.assertEqual((result.completed, result.failed, len(result.outputs)), (2, 1, 4))
        self.assertEqual(len({path.name for path in result.outputs}), 4)
        with zipfile.ZipFile(result.archive) as archive:
            self.assertEqual(len(archive.namelist()), 5)
            self.assertIsNone(archive.testzip())
        with result.report.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertIn("UnidentifiedImageError", rows[2]["错误"])

    def test_cancel_retains_completed_files(self):
        paths = []
        for name in ("1.png", "2.png"):
            path = self.root / name
            sample().save(path)
            paths.append(path)
        should_cancel = [False]

        def progress(fraction, description):
            if fraction >= 0.5:
                should_cancel[0] = True

        def cancel():
            if should_cancel[0]:
                raise Cancelled()

        result = run_batch(paths, self.root / "out", Options(), check_cancel=cancel, progress=progress)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.completed, 1)
        self.assertTrue(result.outputs[0].exists())
        self.assertTrue(result.archive.exists())

    def test_natural_sort_excludes_output_and_hidden_directory_access(self):
        for name in ("10.png", "2.png"):
            sample().save(self.root / name)
        output = self.root / "out"
        output.mkdir()
        sample().save(output / "old.png")
        sources = collect_sources("本机文件夹", None, str(self.root), True, "路径自然排序", excluded_root=output)
        self.assertEqual([path.name for path in sources], ["2.png", "10.png"])
        with self.assertRaisesRegex(ValueError, "hide-ui-dir-config"):
            collect_sources("本机文件夹", None, str(self.root), True, "路径自然排序", False)


if __name__ == "__main__":
    unittest.main()
