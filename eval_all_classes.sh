#!/bin/bash
# Run single-class eval for all remaining VOC classes
CKPT="/root/autodl-tmp/mycheckpoints/best_checkpoint.pt"
CONFIG="config.yaml"
LOG="log/1cls_eval.txt"

CLASSES=(aeroplane bicycle bird boat bottle bus car cat cow dog horse motorbike pottedplant sheep sofa train)

for cls in "${CLASSES[@]}"; do
    echo "===== $cls =====" | tee -a "$LOG"
    python eval.py --checkpoint "$CKPT" --config "$CONFIG" --category "$cls" --max-samples 0 2>&1 | tee -a "$LOG"
    echo "" >> "$LOG"
done

echo "Done. Results saved to $LOG"
