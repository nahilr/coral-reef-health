import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.image_metadata import extract_image_metadata, extract_video_metadata


class ImageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "reef_001.jpg"
        Image.new("RGB", (32, 24), color=(20, 80, 100)).save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def test_no_metadata_returns_no_location(self):
        result = extract_image_metadata(self.image)
        self.assertEqual(result["width"], 32)
        self.assertEqual(result["height"], 24)
        self.assertFalse(result["location"]["available"])
        self.assertEqual(result["location"]["display"], "No location available")

    def test_json_sidecar_location(self):
        sidecar = self.root / "reef_001.json"
        sidecar.write_text(json.dumps({"image": "reef_001.jpg", "site": "Agatti", "atoll": "Lakshadweep"}), encoding="utf-8")
        result = extract_image_metadata(self.image)
        location = result["location"]
        self.assertTrue(location["available"])
        self.assertIn("Agatti", location["display"])
        self.assertIn("Lakshadweep", location["display"])
        self.assertIn("reef_001.json", location["source"])

    def test_image_stem_sidecar_can_contain_location_only(self):
        sidecar = self.root / "reef_001.json"
        sidecar.write_text(json.dumps({"location": "Kavaratti"}), encoding="utf-8")
        result = extract_image_metadata(self.image)
        self.assertTrue(result["location"]["available"])
        self.assertEqual(result["location"]["display"], "Kavaratti")

    def test_csv_manifest_match_by_photo_id_and_coordinates(self):
        manifest = self.root / "labels.csv"
        manifest.write_text(
            "Photo_ID,Site,Latitude,Longitude\nreef_001,Agatti,10.1234,72.4567\nother,Elsewhere,0,0\n",
            encoding="utf-8",
        )
        result = extract_image_metadata(self.image, search_roots=[self.root])
        location = result["location"]
        self.assertTrue(location["available"])
        self.assertIn("Agatti", location["display"])
        self.assertAlmostEqual(location["latitude"], 10.1234)
        self.assertAlmostEqual(location["longitude"], 72.4567)

    def test_uploaded_single_record_metadata(self):
        sidecar = self.root / "sidecar.yaml"
        sidecar.write_text("island: Kadmat\n", encoding="utf-8")
        result = extract_image_metadata(self.image, label_path=sidecar)
        self.assertTrue(result["location"]["available"])
        self.assertEqual(result["location"]["display"], "Kadmat")

    def test_video_label_metadata_supplies_location(self):
        video = self.root / "transect_01.mp4"
        video.write_bytes(b"not a real video; ffprobe may reject this fixture")
        sidecar = self.root / "video_location.json"
        sidecar.write_text(
            json.dumps({"video": "transect_01.mp4", "site": "Agatti", "latitude": 10.1234, "longitude": 72.4567}),
            encoding="utf-8",
        )
        result = extract_video_metadata(video, label_path=sidecar)
        location = result["location"]
        self.assertTrue(location["available"])
        self.assertIn("Agatti", location["display"])
        self.assertAlmostEqual(location["latitude"], 10.1234)
        self.assertAlmostEqual(location["longitude"], 72.4567)

    def test_video_without_location_returns_clear_fallback(self):
        video = self.root / "no_gps_video.mp4"
        video.write_bytes(b"not a real video; ffprobe may reject this fixture")
        result = extract_video_metadata(video)
        self.assertFalse(result["location"]["available"])
        self.assertEqual(result["location"]["display"], "No location available")


if __name__ == "__main__":
    unittest.main()
