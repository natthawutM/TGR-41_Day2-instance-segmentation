# TGR-41 Day 2 | Instance IoU experiments

วัดด้วย ONNX Runtime CPU บน Windows และ COCO Valid 20 ภาพเดิม (192 instances) เลือกค่า watershed จากผลรวม Instance IoU + Counting แบบจับคู่ ค่า IoU 0.90 ในตารางคือจำนวนคู่ที่มี IoU ถึงเกณฑ์

| Variant | Instance IoU | Counting แบบจับคู่ | คู่ IoU ≥0.90 | คลาดเคลื่อน (ชิ้น/ภาพ) |
|---|---:|---:|---:|---:|
| `fp32_512` baseline | 85.43% | 98.09% | 52 | 0.15 |
| `fp32_512_tuned` หนึ่งรอบ | 86.27% | 98.59% | 57 | 0.10 |
| `fp32_640_flip2` สองทิศทาง | 88.51% | 98.74% | 97 | 0.15 |
| `fp32_512_flip4` สี่ทิศทาง | **90.52%** | **98.71%** | **135** | **0.10** |

ทั้งสามรุ่นใช้ checkpoint เดิม Flip TTA เฉลี่ย logits foreground และ boundary หลังกลับพิกัดของแต่ละภาพ แล้วทำ watershed ครั้งเดียว 512 flip4 ใช้ foreground 0.38, boundary 0.12 และ marker separation 24 px ภาพ Shell `sep_487a` เปลี่ยนจาก 8/10 เป็น 10/10 และ IoU จาก 61.58% เป็น 88.14%

ทดลอง 768 แบบสี่ทิศทางได้ 89.36% IoU และทดลอง fine-tune 640 เพิ่ม 12 epochs ได้ 83.41% IoU หลังจูน threshold ชุดเดียวกัน การจูนและการเลือกใช้ Valid ชุดเดียวกัน ผลจึงเป็น validation และยังต้องประเมินซ้ำกับชุดภาพอิสระ

## วัดบน Pi

ใช้ holdout 8 ภาพเดิมเพื่อเทียบ baseline 512 ที่เคยวัดได้ 0.772 s/ภาพ

```bash
for variant in fp32_512_flip4 fp32_640_flip2 fp32_512_tuned; do
  python infer.py --input test_holdout --weights weights --variant "$variant" \
    --out "bench_$variant" --threads 4 --warmup 5 || break
done
```

เลือก variant ที่ IoU สูงสุดในรุ่นที่ยังวัดได้ไม่เกิน 1.00 s/ภาพ ผล Windows ใช้เลือก candidate และเวลา Pi ของ variant ใหม่ยังรอวัด
