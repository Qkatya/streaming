# Simple Instructions - Edit Concatenated GT Peaks

## ✅ AUTOMATIC SAVING (Easiest Way)

### Just run your analysis as normal:

```bash
python blinking_split_analysis.py
```

**The script now automatically saves concatenated data!**

At the end, you'll see:
```
SAVING CONCATENATED DATA FOR PEAK EDITOR
✓ Saved concatenated signals to: concatenated_signals_20231203_143052.pkl
  
  To edit GT peaks, run:
    python launch_peak_editor_from_analysis.py concatenated_signals_20231203_143052.pkl
```

### Then edit your peaks:

```bash
python launch_peak_editor_from_analysis.py concatenated_signals_20231203_143052.pkl
```

**Each run creates a new timestamped file, so nothing gets overwritten!**

---

## 📁 Files Created

Each time you run `blinking_split_analysis.py`, it creates:
- `concatenated_signals_YYYYMMDD_HHMMSS.pkl` (unique timestamp)

Example:
- `concatenated_signals_20231203_143052.pkl`
- `concatenated_signals_20231203_150234.pkl`
- `concatenated_signals_20231203_163015.pkl`

**No overwriting! Each analysis is saved separately.**

---

## 🎯 What You Get After Editing:

After editing and clicking "Save Changes":
- `final_gt_peaks.pkl` - Your edited GT peaks (frame indices)

Use them like this:
```python
import pickle
with open('final_gt_peaks.pkl', 'rb') as f:
    edited_gt_peaks = pickle.load(f)

print(f"Total edited GT peaks: {len(edited_gt_peaks)}")
```

---

## 🔄 Complete Workflow

```bash
# 1. Run your analysis (automatically saves concatenated data)
python blinking_split_analysis.py

# 2. Copy the filename from the output, then run:
python launch_peak_editor_from_analysis.py concatenated_signals_20231203_143052.pkl

# 3. Edit peaks in browser at http://127.0.0.1:8050
#    - Click on peaks to remove
#    - Click elsewhere to add
#    - Save changes

# 4. Use edited peaks in your analysis
python
>>> import pickle
>>> with open('final_gt_peaks.pkl', 'rb') as f:
...     peaks = pickle.load(f)
```

---

## 💡 Tips

- **Each analysis run creates a new file** - no overwriting!
- **Timestamped filenames** make it easy to track different runs
- **You can edit any saved file** anytime by specifying its name
- **The latest file** can be found by sorting by timestamp

---

## 🎉 That's It!

Just run `python blinking_split_analysis.py` and it will automatically save the concatenated data for you!
