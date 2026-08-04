# Method_Transformer audit — 2026-08-05

## Executive finding

ผล `run_33_evaluate` บน HouseGAN ใช้เป็นผลวิจัยไม่ได้ เพราะ checkpoint มาจาก
`run_33` ซึ่งฝึกด้วย `Topo_bottleneck` แต่ถูกนำไปทดสอบกับ `Topo_HouseGAN`.
ไฟล์ weights ทั้งสองตำแหน่งมี SHA-256 เดียวกัน:

`571189501B5D70EA98AF59718EC6F45646A55FAA6AE70437B75F0742DF5875C0`

ภาพที่แสดง floorplan เพียงครึ่งเดียวเป็นปัญหาอีกชั้นหนึ่งใน renderer: axis limits
ถูกคำนวณจากช่วงของ predicted trajectories แทนที่จะใช้ bounds ของ geometry ทั้งผัง.
ดังนั้นภาพเดิมผสมทั้ง visualization bug และ invalid cross-topology evaluation.

## Problems found

1. **Checkpoint/dataset mismatch** — ไม่มี provenance อยู่ใน checkpoint รุ่นเก่าและ
   test script ไม่ตรวจ topology ก่อนประเมิน.
2. **Canvas cropping** — `xlim`/`ylim` ใช้ prediction bounds ทำให้ห้องหรือ exit ที่
   rollout ไปไม่ถึงถูกตัดออก.
3. **Metrics in incomparable units** — ADE/FDE ถูกเฉลี่ยใน normalized coordinates
   ทั้งที่แต่ละ plan มี `meta["scale"]` ต่างกัน. Primary metrics ต้องแปลงเป็น metres
   ต่อ sample ก่อน aggregation.
4. **Spatial information bottleneck** — geometry CNN ใช้ global average pooling
   จาก 4x4 เหลือ 1x1 จึงทิ้งตำแหน่งเชิงพื้นที่ของห้องและประตูจำนวนมาก.
5. **Train/rollout contract weakness** — absolute-coordinate teacher forcing ให้
   one-step loss ต่ำมาก แต่ autoregressive rollout สะสม error ได้มาก.
6. **No wall constraint** — loss เดิมไม่ลงโทษ predicted points ที่อยู่ในผนังหรือ
   นอก walkable region.
7. **Padding ambiguity** — `(0, 0)` ถูกใช้โดยอ้อมเป็น padding sentinel ทั้งที่เป็น
   normalized coordinate ที่ถูกต้องได้; ต้อง mask จาก sequence lengths.
8. **Invalid helper evaluators** — `run_housegan_evaluations_normalized.py` สร้าง
   A* paths เหมือนกันทุกโมเดลโดยไม่ execute checkpoint และ
   `run_authentic_evaluations.py` วาด seeded synthetic curves หลังเพียงโหลด state dict.
   ทั้งสองเส้นทางถูก disable เพื่อไม่ให้สร้างหลักฐานวิจัยปลอมเพิ่ม.

## Implemented contract (new checkpoints)

- Spatial 4x4 geometry encoder (`geo_encoder_type="spatial"`).
- Bounded delta prediction (`prediction_mode="delta"`) เพื่อให้ training และ rollout
  ใช้ transition contract เดียวกัน.
- Differentiable walkability/collision loss sampled จาก occupancy grid.
- Explicit length mask.
- Checkpoint format v2 เก็บ model config, data config, normalization name, dataset
  identity และ epoch พร้อม weights.
- Test-time fail-fast เมื่อ dataset ใน checkpoint ไม่ตรงกับ dataset ที่ขอประเมิน;
  override ได้เฉพาะ debug ด้วย `--allow-dataset-mismatch`.
- ADE/FDE รายงานเป็น metres; normalized values เก็บเป็น secondary diagnostics.
- เพิ่ม wall-violation rate และ goal-success rate.
- Full-floorplan plot ใช้ geometry bounds และแสดง GT เป็นเส้นประจางเพื่อวินิจฉัย.

## Required retraining

การแก้ renderer ทำให้เห็นผังเต็มทันที แต่ไม่สามารถทำให้ weights ของ bottleneck
กลายเป็น HouseGAN weights ได้. ต้อง train ใหม่ด้วย config ปัจจุบัน:

```bash
cd AI_GenerateTimeseries/AI_Train/Method_Transformer
python train_transformer.py --config config_train.json
```

จากนั้นทดสอบ checkpoint ของ run ใหม่โดยไม่ใช้ mismatch override:

```bash
python test_transformer.py \
  --config config_test.json \
  --model_path ../../AI_Result/Method_Transformer/outputs/run_NEW/weights/best_model.pth \
  --run_path ../../AI_Result/Method_Transformer/outputs/run_NEW
```

## Recommended ablation plan

ใช้ split และ random seeds ชุดเดียวกันอย่างน้อย 3 seeds ต่อ condition:

| ID | Geometry encoder | Prediction | Collision loss | Purpose |
|---|---|---|---:|---|
| A | pooled | absolute | 0.00 | legacy baseline |
| B | spatial | absolute | 0.00 | isolate spatial preservation |
| C | spatial | delta | 0.00 | isolate rollout contract |
| D | spatial | delta | 0.05 | proposed full model |

รายงาน ADE/FDE (m), wall-violation rate, goal-success rate, inference latency และ
confidence interval แยกตาม single/half/full occupancy. ห้ามรวมผล debug mismatch
เข้าตารางหรือ UI performance comparison.

## Diagnostic result for the reported case

`plan_44_fd18_100042_00_half` ผ่าน normalization round-trip และ trajectory ทุกจุด
อยู่ใน shared `[0,1]` frame. เมื่อใช้ old mismatched run เพื่อ debug เท่านั้น:

- ADE: 7.1437 m
- FDE: 13.2302 m
- Goal success: 0/6 agents

ภาพผังเต็มถูกเก็บแยกใน
`run_33_evaluate_debug_mismatch/test_results/predictions/` และติดหัวข้อ
`DEBUG ONLY — checkpoint/dataset mismatch` ชัดเจน.
