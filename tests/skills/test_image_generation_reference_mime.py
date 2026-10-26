"""Gemini reference images must be labelled with the MIME type of the bytes sent.

The skill forwards reference bytes unconverted, so a fixed ``image/jpeg`` label
misdescribes every PNG/WebP/GIF reference. These tests stay offline: HTTP is
mocked and the reference images are generated with Pillow into ``tmp_path``.
"""

import base64
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_loader import FakeResp, load  # noqa: E402

img = load("image-generation")

GEMINI_OK = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(b"out").decode(),
                        }
                    }
                ]
            }
        }
    ]
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in [
        "GEMINI_API_KEY",
        "MINIMAX_API_KEY",
        "IMAGE_GENERATION_PROVIDER",
        "IMAGE_GENERATION_API_KEY",
        "IMAGE_GENERATION_BASE_URL",
        "IMAGE_GENERATION_MODEL",
        "IMAGE_GENERATION_SIZE",
    ]:
        monkeypatch.delenv(key, raising=False)


def _write_image(path: Path, fmt: str) -> Path:
    Image.new("RGB", (2, 2), (10, 20, 30)).save(path, format=fmt)
    return path


def _capture_gemini_post(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return FakeResp(GEMINI_OK)

    monkeypatch.setattr(img.requests, "post", fake_post)
    return captured


@pytest.mark.parametrize(
    "filename,fmt,expected",
    [
        pytest.param("reference.png", "PNG", "image/png", id="png"),
        pytest.param("reference.PNG", "PNG", "image/png", id="uppercase-png"),
        pytest.param("reference.webp", "WEBP", "image/webp", id="webp"),
        pytest.param("reference.WEBP", "WEBP", "image/webp", id="uppercase-webp"),
        pytest.param("reference.gif", "GIF", "image/gif", id="gif"),
        pytest.param("reference.GIF", "GIF", "image/gif", id="uppercase-gif"),
        pytest.param("reference.jpg", "JPEG", "image/jpeg", id="jpg"),
        pytest.param("reference.jpeg", "JPEG", "image/jpeg", id="jpeg"),
    ],
)
def test_reference_mime_type_matches_the_submitted_bytes(
    monkeypatch, tmp_path, filename, fmt, expected
):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    captured = _capture_gemini_post(monkeypatch)

    reference = _write_image(tmp_path / filename, fmt)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a red apple", encoding="utf-8")

    img.generate_image(
        str(prompt_file), [str(reference)], str(tmp_path / "out.png"), "1:1"
    )

    parts = captured["json"]["contents"][0]["parts"]
    inline = [part for part in parts if "inlineData" in part]
    assert len(inline) == 1
    assert inline[0]["inlineData"]["mimeType"] == expected
    # The declared type must describe the bytes actually sent, not a conversion.
    assert base64.b64decode(inline[0]["inlineData"]["data"]) == reference.read_bytes()


def test_mixed_reference_formats_keep_their_own_labels_and_order(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    captured = _capture_gemini_post(monkeypatch)

    first = _write_image(tmp_path / "first.png", "PNG")
    second = _write_image(tmp_path / "second.webp", "WEBP")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("scene", encoding="utf-8")

    img.generate_image(
        str(prompt_file), [str(first), str(second)], str(tmp_path / "out.png"), "1:1"
    )

    inline = [
        part["inlineData"]
        for part in captured["json"]["contents"][0]["parts"]
        if "inlineData" in part
    ]
    assert [entry["mimeType"] for entry in inline] == ["image/png", "image/webp"]
    assert base64.b64decode(inline[0]["data"]) == first.read_bytes()
    assert base64.b64decode(inline[1]["data"]) == second.read_bytes()


def test_generation_without_references_does_not_import_pillow(monkeypatch, tmp_path):
    """A reference-free call must keep working when Pillow is unavailable."""
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    _capture_gemini_post(monkeypatch)
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a red apple", encoding="utf-8")
    out = tmp_path / "out.png"

    message = img.generate_image(str(prompt_file), [], str(out), "1:1")

    assert out.read_bytes() == b"out"
    assert "Successfully generated image" in message
