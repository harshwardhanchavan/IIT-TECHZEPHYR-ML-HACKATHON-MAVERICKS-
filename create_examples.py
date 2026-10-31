"""Create simple test sketches in ./examples (no dataset needed).   python create_examples.py"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "examples"
W = 6


def canvas():
    im = Image.new("RGB", (600, 600), "white")
    return im, ImageDraw.Draw(im)


def mug():
    im, d = canvas()
    d.rounded_rectangle((150, 180, 400, 440), 25, outline="black", width=W)
    d.arc((340, 230, 500, 390), -90, 90, fill="black", width=W)
    d.ellipse((150, 160, 400, 210), outline="black", width=W)
    return im


def chair():
    im, d = canvas()
    d.rectangle((180, 100, 420, 300), outline="black", width=W)
    d.rectangle((160, 300, 440, 340), outline="black", width=W)
    for x in (170, 410):
        d.rectangle((x, 340, x + 25, 520), outline="black", width=W)
    return im


def car():
    im, d = canvas()
    d.rounded_rectangle((100, 280, 500, 380), 30, outline="black", width=W)
    d.line([(190, 280), (240, 210), (380, 210), (430, 280)], fill="black", width=W)
    d.line([(240, 210), (240, 280)], fill="black", width=W)
    d.line([(380, 210), (380, 280)], fill="black", width=W)
    for cx in (190, 410):
        d.ellipse((cx - 40, 340, cx + 40, 420), outline="black", width=W)
    return im


def house():
    im, d = canvas()
    d.rectangle((150, 270, 450, 480), outline="black", width=W)
    d.line([(130, 270), (300, 130), (470, 270), (130, 270)], fill="black", width=W)
    d.rectangle((270, 370, 340, 480), outline="black", width=W)
    d.rectangle((180, 310, 240, 370), outline="black", width=W)
    return im


def lamp():
    im, d = canvas()
    d.polygon([(200, 120), (400, 120), (460, 280), (140, 280)], outline="black")
    d.line([(200, 120), (400, 120), (460, 280), (140, 280), (200, 120)], fill="black", width=W)
    d.line([(300, 280), (300, 460)], fill="black", width=W + 4)
    d.ellipse((220, 460, 380, 500), outline="black", width=W)
    return im


def bottle():
    im, d = canvas()
    d.line([(270, 100), (270, 190), (220, 260), (220, 480)], fill="black", width=W)
    d.line([(330, 100), (330, 190), (380, 260), (380, 480)], fill="black", width=W)
    d.line([(270, 100), (330, 100)], fill="black", width=W)
    d.arc((220, 450, 380, 510), 0, 180, fill="black", width=W)
    d.line([(235, 330), (365, 330)], fill="black", width=W)
    return im


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for fn in (mug, chair, car, house, lamp, bottle):
        fn().save(OUT / f"{fn.__name__}.png")
    print("Wrote", ", ".join(f"{n}.png" for n in ("mug", "chair", "car", "house", "lamp", "bottle")), "to", OUT)
