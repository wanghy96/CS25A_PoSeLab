from PIL import Image

img = Image.open("icon_clean.png").convert("RGBA")

sizes = [(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]

img.save(
    "PoseLab.ico",
    format="ICO",
    sizes=sizes
)

print("✅ PoseLab.ico 已生成")
