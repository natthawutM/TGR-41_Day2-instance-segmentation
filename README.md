# TGR-41 Day 2 Coffee Bean Instance Segmentation

ชุดส่งงานและรันบน Raspberry Pi 5 ใช้รุ่น A ที่เทรนเต็ม 40 epochs และจูน watershed แล้ว MobileNetV2–U-Net ทำนาย foreground และ boundary และใช้ classifier Day 1 จัดคลาสให้ทุก ID ทั้งสองโมเดลทำงานผ่าน ONNX Runtime CPU

## รันบน Raspberry Pi 5

ใช้ Raspberry Pi OS 64-bit (`uname -m` แสดง `aarch64`) เปิด Terminal ในโฟลเดอร์นี้ และวางภาพ test จริงไว้ใน `test/` โดยให้ชื่อไฟล์ก่อน extension ไม่ซ้ำกัน

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python infer.py --input test --weights weights --out outputs --threads 4 --warmup 5
```

ถ้าเครื่องไม่มีโมดูล `venv` ให้ติดตั้ง `python3-venv` ผ่าน `apt` ก่อน ข้อมูลอ้างอิง [Raspberry Pi OS](https://www.raspberrypi.com/documentation/computers/os.html#use-python-on-a-raspberry-pi) · [ONNX Runtime CPU และ Linux ARM64](https://onnxruntime.ai/docs/get-started/with-python)

ไฟล์ที่ใช้รันอยู่ใน `weights/` ได้แก่ `seg_fp32_512.onnx`, `classifier.onnx` และ `manifest.json` พร้อม `infer.py`, `data.py`, `post.py` และ dependencies ใน `requirements.txt`

## ไฟล์ส่งงาน

- `outputs/seg_mask/<ชื่อภาพ>.png` เป็น PNG 16-bit ขนาดเท่าภาพต้นฉบับ พื้นหลัง 0 และ instance ID 1..N ทุกภาพอยู่ในโฟลเดอร์เดียวกัน
- `outputs/predictions.json` มี `image_name` และ `objects` ที่เก็บ `id`, `class`, `polygon` หนึ่งรายการต่อ ID โดย polygon ใช้ contour ภายนอกที่ใหญ่ที่สุดของ ID นั้น ส่วนที่แยกจาก contour หลักยังอยู่ใน PNG
- `outputs/vis/` มี overlay พร้อม ID และคลาส
- `outputs/benchmark.json` บันทึก median วินาที/ภาพแบบ end-to-end รวม preprocessing, model, watershed, classifier, polygon และเขียนผลหลัง warm-up

ในชุดนี้ `outputs/` เป็นโครงสร้างว่าง ให้สร้างผลด้วยภาพ test ของผู้จัดก่อนส่ง และใช้โฟลเดอร์ output ว่างทุกครั้งเมื่อเปลี่ยนชุดภาพ เพื่อไม่ให้ PNG จากรอบก่อนหน้าปะปน ควรอัปโหลดทุกไฟล์ในโฟลเดอร์นี้ รวม `weights/best.pt` และ ONNX ทั้งสองไฟล์ ใช้ `test/` และ `.venv/` บนเครื่องที่รัน

ผลบน Valid 20 ภาพจาก COCO export Stretch 432×432 ด้วย evaluator ของโปรเจกต์ได้ Instance IoU **85.43%** และ Counting **98.55%** เวลาที่วัดแล้วบน Windows CPU คือ **0.069 วินาที/ภาพ** และยังไม่มีผลวัดบน Pi 5

## เทรนใหม่หรือประเมิน COCO

ใช้ชุด COCO instance segmentation ที่มี `train/_annotations.coco.json` และ `valid/_annotations.coco.json` ติดตั้ง `requirements-train.txt` บนเครื่องเทรน และใช้ PyTorch รุ่น CUDA เมื่อต้องการเทรนบน NVIDIA GPU

```bash
python -m pip install -r requirements-train.txt
python train.py --data /path/to/coco --config configs/full_A.json --out runs/A
```

เพิ่ม `--allow-stretch` เมื่อใช้ export ที่เปิด Stretch ใช้ `configs/full_B.json` สำหรับรัน Copy-Paste ค่า watershed ที่เลือกเก็บใน `configs/watershed_selected.json` และฝังอยู่ใน `weights/manifest.json` สำหรับ inference แล้ว
