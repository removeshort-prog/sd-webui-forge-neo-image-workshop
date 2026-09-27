import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from forge_image_workshop.censor import (
    ANIME_DEFAULT_TARGETS, CensorOptions, FrameCensor, build_mask, media_sources,
    process_gif, process_still, render_censor, run_censor_batch,
)
from forge_image_workshop.engine import Cancelled


class FakeDetector:
    def __init__(self, boxes_by_call=None):
        self.calls = 0
        self.boxes_by_call = boxes_by_call or [[(8, 6, 24, 20)]]

    def detect(self, image, targets, confidence):
        boxes = self.boxes_by_call[min(self.calls, len(self.boxes_by_call) - 1)]
        self.calls += 1
        return list(boxes)


class CensorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_mask_shapes_and_only_masked_pixels_change(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(50, dtype=np.uint8)
        for shape in ("rect", "ellipse", "fit"):
            mask = build_mask(image, [(10, 10, 30, 28)], shape)
            self.assertGreater(int(mask.sum()), 0)
            options = CensorOptions(shape=shape, mode="mosaic", strength=20, dilate_px=0)
            output, _ = render_censor(image, mask, options)
            self.assertTrue(np.array_equal(output[mask == 0], image[mask == 0]))
            self.assertGreater(int(np.abs(output.astype(int) - image.astype(int)).sum()), 0)

    def test_frame_interval_and_hold(self):
        detector = FakeDetector([[(8, 6, 24, 20)], [], []])
        options = CensorOptions(detect_every=2, hold=1, shape="rect", dilate_px=0)
        censor = FrameCensor(detector, options, ANIME_DEFAULT_TARGETS)
        frames = []
        for i in range(5):
            frame = np.zeros((32, 40, 3), dtype=np.uint8)
            frame[:, :, 0] = np.arange(40, dtype=np.uint8)[None, :]
            frame[:, :, 1] = 30 + i
            frames.append(frame)
        outputs = [censor(frame)[0] for frame in frames]
        self.assertEqual(detector.calls, 3)
        self.assertEqual(censor.hit_frames, 4)
        self.assertFalse(np.array_equal(outputs[0], frames[0]))
        self.assertTrue(np.array_equal(outputs[4], frames[4]))

    def test_transparent_still_keeps_alpha_and_reports_boxes(self):
        source = self.root / "透明图.png"
        image = Image.new("RGBA", (40, 30), (240, 20, 30, 0))
        image.putalpha(Image.new("L", image.size, 127))
        image.save(source)
        output = self.root / "result.webp"
        stats = process_still(source, output, FakeDetector(), CensorOptions(shape="rect", dilate_px=0), ANIME_DEFAULT_TARGETS, lambda: None)
        self.assertEqual(stats["boxes"], 1)
        with Image.open(output) as read:
            self.assertEqual(read.mode, "RGBA")
            self.assertEqual(read.getchannel("A").getextrema(), (127, 127))

    def test_gif_preserves_duration_and_loop(self):
        source = self.root / "input.gif"
        frames = [Image.new("RGBA", (32, 24), (20 + i * 10, 80, 120, 180)) for i in range(3)]
        frames[0].save(source, save_all=True, append_images=frames[1:], duration=[40, 70, 90], loop=3)
        output = self.root / "output.gif"
        options = CensorOptions(shape="rect", dilate_px=0)
        censor = FrameCensor(FakeDetector(), options, ANIME_DEFAULT_TARGETS)
        stats = process_gif(source, output, censor, lambda: None)
        self.assertEqual(stats["frames"], 3)
        with Image.open(output) as read:
            self.assertEqual(read.info.get("loop"), 3)
            self.assertEqual([frame.info.get("duration") for frame in ImageSequence.Iterator(read)], [40, 70, 90])

    def test_batch_creates_report_zip_and_continues_after_bad_media(self):
        good = self.root / "good.png"
        Image.new("RGB", (32, 24), "red").save(good)
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image")
        result = run_censor_batch([good, bad], self.root / "out", CensorOptions(shape="rect", dilate_px=0),
                                  detector=FakeDetector())
        self.assertEqual((result.completed, result.failed), (1, 1))
        self.assertTrue(result.report.exists())
        self.assertTrue(result.archive.exists())
        with zipfile.ZipFile(result.archive) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 2)
        with result.report.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)

    def test_cancel_does_not_delete_completed_media(self):
        source = self.root / "one.png"
        second = self.root / "two.png"
        Image.new("RGB", (32, 24), "red").save(source)
        Image.new("RGB", (32, 24), "blue").save(second)
        cancelled = [False]

        def check_cancel():
            if cancelled[0]:
                raise Cancelled()

        def progress(fraction, description):
            if fraction >= 0.5:
                cancelled[0] = True

        result = run_censor_batch([source, second], self.root / "out", CensorOptions(shape="rect", dilate_px=0),
                                  check_cancel=check_cancel, progress=progress, detector=FakeDetector())
        self.assertTrue(result.cancelled)
        self.assertEqual(result.completed, 1)
        self.assertEqual(len(result.outputs), 1)
        self.assertTrue(result.report.exists())

    def test_media_sort_and_directory_exclusion(self):
        input_dir = self.root / "input"
        input_dir.mkdir()
        output_dir = input_dir / "results"
        output_dir.mkdir()
        for name in ("10.png", "2.gif"):
            (input_dir / name).write_bytes(b"x")
        (output_dir / "old.mp4").write_bytes(b"x")
        files = media_sources("本机文件夹", None, input_dir, True, "文件名自然排序", True, output_dir)
        self.assertEqual([path.name for path in files], ["2.gif", "10.png"])
        with self.assertRaisesRegex(ValueError, "hide-ui-dir-config"):
            media_sources("本机文件夹", None, input_dir, True, "路径自然排序", False)


if __name__ == "__main__":
    unittest.main()
