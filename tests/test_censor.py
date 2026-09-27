import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from forge_image_workshop.censor import (
    ANIME_DEFAULT_TARGETS,
    CensorOptions,
    build_mask,
    media_sources,
    process_still,
    render_censor,
    run_censor_batch,
    targets_for,
)
from forge_image_workshop.engine import Cancelled


class FakeDetector:
    def __init__(self, boxes=None):
        self.boxes = boxes or [(8, 6, 24, 20)]

    def detect(self, image, targets, confidence):
        return list(self.boxes)


class CensorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_anime_targets_are_fixed_and_extra_targets_are_filtered(self):
        options = CensorOptions(extra_targets=("nipple_f", "real_only"))
        self.assertEqual(targets_for(options), [*ANIME_DEFAULT_TARGETS, "nipple_f"])

    def test_mask_shapes_only_change_masked_pixels(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(50, dtype=np.uint8)
        for shape in ("rect", "ellipse", "fit"):
            with self.subTest(shape=shape):
                mask = build_mask(image, [(10, 10, 30, 28)], shape)
                output, _ = render_censor(image, mask, CensorOptions(shape=shape, dilate_px=0, strength=20))
                self.assertGreater(int(mask.sum()), 0)
                self.assertTrue(np.array_equal(output[mask == 0], image[mask == 0]))
                self.assertGreater(int(np.abs(output.astype(int) - image.astype(int)).sum()), 0)

    def test_transparent_still_keeps_alpha_and_reports_boxes(self):
        source = self.root / "透明图.png"
        image = Image.new("RGBA", (40, 30), (240, 20, 30, 0))
        image.putalpha(Image.new("L", image.size, 127))
        image.save(source)
        output = self.root / "result.webp"
        stats = process_still(source, output, FakeDetector(), CensorOptions(shape="rect", dilate_px=0),
                              ANIME_DEFAULT_TARGETS, lambda: None)
        self.assertEqual(stats["boxes"], 1)
        with Image.open(output) as read:
            self.assertEqual(read.mode, "RGBA")
            self.assertEqual(read.getchannel("A").getextrema(), (127, 127))

    def test_animated_image_is_rejected(self):
        source = self.root / "input.gif"
        Image.new("RGB", (20, 20), "red").save(
            source, save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")], duration=100
        )
        with self.assertRaisesRegex(ValueError, "静态单帧"):
            process_still(source, self.root / "output.gif", FakeDetector(), CensorOptions(),
                          ANIME_DEFAULT_TARGETS, lambda: None)

    def test_batch_creates_report_zip_and_continues_after_bad_image(self):
        good = self.root / "good.png"
        Image.new("RGB", (32, 24), "red").save(good)
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image")
        result = run_censor_batch([good, bad], self.root / "out", CensorOptions(shape="rect", dilate_px=0),
                                  detector=FakeDetector())
        self.assertEqual((result.completed, result.failed), (1, 1))
        with zipfile.ZipFile(result.archive) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 2)
        with result.report.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)

    def test_cancel_keeps_completed_images(self):
        paths = []
        for name, color in (("one.png", "red"), ("two.png", "blue")):
            path = self.root / name
            Image.new("RGB", (32, 24), color).save(path)
            paths.append(path)
        cancelled = [False]

        def check_cancel():
            if cancelled[0]:
                raise Cancelled()

        def progress(fraction, description):
            if fraction >= 0.5:
                cancelled[0] = True

        result = run_censor_batch(paths, self.root / "out", CensorOptions(shape="rect", dilate_px=0),
                                  check_cancel=check_cancel, progress=progress, detector=FakeDetector())
        self.assertTrue(result.cancelled)
        self.assertEqual(result.completed, 1)
        self.assertTrue(result.outputs[0].exists())
        self.assertTrue(result.report.exists())

    def test_image_sort_excludes_output_and_rejects_disabled_directories(self):
        input_dir = self.root / "input"
        input_dir.mkdir()
        output_dir = input_dir / "results"
        output_dir.mkdir()
        for name in ("10.png", "2.png"):
            Image.new("RGB", (8, 8), "white").save(input_dir / name)
        Image.new("RGB", (8, 8), "black").save(output_dir / "old.png")
        files = media_sources("本机文件夹", None, input_dir, True, "文件名自然排序", True, output_dir)
        self.assertEqual([path.name for path in files], ["2.png", "10.png"])
        with self.assertRaisesRegex(ValueError, "hide-ui-dir-config"):
            media_sources("本机文件夹", None, input_dir, True, "路径自然排序", False)


if __name__ == "__main__":
    unittest.main()
