# TGR-41 Day 2 | FP32 512 submission

สำเนาสำหรับส่งงานของ MobileNetV2–U-Net รุ่น A ที่เทรน 40 epochs พร้อมค่า watershed ที่เลือก และ classifier Day 1 สำหรับ 7 คลาส ใช้ `weights/seg_fp32_512.onnx`, `weights/classifier.onnx` และ `weights/manifest.json` มี `weights/best.pt` เป็น checkpoint การเทรน

## รันบน Raspberry Pi ด้วย CPU

บน Raspberry Pi OS 64-bit ให้เปิด Terminal ในโฟลเดอร์นี้ และวางภาพ Test ของผู้จัดใน `test/` โดยตรง

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python infer.py --input test/ --weights weights/ --out outputs/
```

ตัวรันโหลด ONNX จากดิสก์ผ่าน CPUExecutionProvider หลังปรับ EXIF และแปลงเป็น RGB จะ resize แบบรักษาสัดส่วนและ pad เป็น 512×512 แล้ว normalize ด้วย mean [0.485, 0.456, 0.406] และ std [0.229, 0.224, 0.225] จากนั้นใช้ watershed แยก ID ตัด padding ออก และคืน mask เป็นขนาดภาพหลังปรับ EXIF แบบ nearest-neighbor ตัวจัดคลาสรับ crop จากกรอบของ ID ขนาด 224×224 และ normalize ตาม manifest การรันไม่มี TTA และใช้ weights ที่มากับโฟลเดอร์

- `outputs/seg_mask/<stem>.png` เป็น PNG 16-bit ขนาดเท่าภาพหลังปรับทิศตาม EXIF พื้นหลัง 0 และ ID 1..N ทุกภาพอยู่ในโฟลเดอร์เดียวกัน
- `outputs/predictions.json` มีหนึ่งรายการต่อภาพพร้อมนามสกุล แต่ละ `objects` มี `id` ตรงกับ PNG, `class` และ `polygon` เป็นพิกัด `[x,y]` ในภาพเต็ม สร้างด้วย `cv2.findContours` โดยเก็บจุดขอบทั้งหมด และตรวจโดยวาด polygon กลับเป็น mask แล้วเทียบทุกพิกเซล
- `outputs/vis/` เก็บ overlay ขนาดด้านยาวไม่เกิน 1280 px ตามค่าเริ่มต้น ใช้ `--vis-max-side 0` หากต้องการภาพ preview ขนาดเต็ม
- `outputs/benchmark.json` เก็บ median เวลาประมวลผลต่อภาพสำหรับการทดสอบของทีม เวลานี้ไม่รวมการโหลดโมเดล

โฟลเดอร์ `outputs/` เตรียมไว้ให้ใส่ผลจากภาพ Test ของผู้จัด ควรรันในโฟลเดอร์ output ว่างเมื่อเปลี่ยนชุดภาพ

ถ้า mask ของ ID เดียวมีหลายส่วน จะเก็บส่วนเชื่อมต่อที่มีพิกเซลมากสุด ถ้ามีรูจะเติมเฉพาะพื้นที่ที่ไม่ทับ instance อื่น และลองเปิดช่องแคบถึงขอบหากรูมีเมล็ดอื่นอยู่ข้างใน เมื่อไม่เหลือรูปที่สร้าง polygon ได้โดยไม่ทับกัน จะละ instance นั้นและจัด ID ใหม่ตามลำดับ โปรแกรมแจ้งชื่อภาพ, ID และจำนวนพิกเซลที่ปรับ หรือแจ้งรายการที่ละทาง stderr PNG, polygon และ overlay ใช้ mask หลังปรับชุดเดียวกัน

สำหรับตรวจคะแนนกับ COCO ที่มีเฉลย ให้ติดตั้ง `pycocotools` เพิ่มแล้วใช้ `python evaluate.py --data coco_test --predictions outputs --out metrics.json` ผลมี F1 ที่ mask IoU ≥ 0.95 ทั้งแบบใช้คลาสและแบบไม่ใช้คลาส แยกไว้ใน `summary.f1_iou_0.95`

โมเดลก่อนปรับการส่งออก polygon วัดบน Pi 5 ได้ IoU 85.41%, Counting แบบจับคู่ 98.09% และ 0.281 s/ภาพ บน Valid 20 ภาพ ส่วน holdout 8 ภาพได้ 0.772 s/ภาพ ตัวเลขทั้งสองชุดเป็นผลอ้างอิง ยังไม่ได้วัดเวลาชุดส่งงานนี้บน Pi 4
