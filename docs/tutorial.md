# PyCrackQ Tutorial

## 1. Installation

Install Python 3 and Git, then run:

```bash
git clone https://github.com/DZ-LT1/PyCrackQ.git
cd PyCrackQ
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat

# macOS or Linux
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Running PyCrackQ

From the PyCrackQ directory, activate the virtual environment and run:

```bash
python main.py
```

The PyCrackQ graphical interface will open.

## 3. Reproducing the Example Results

Use `Examples/5.jpeg` and the fixed parameters below to reproduce the provided example results.

1. Click **Open Image** and select `Examples/5.jpeg`.
2. Turn off **Auto**.
3. Set the image-processing parameters:
   - Binarization method: **Sauvola**
   - Window: **43**
   - k: **0.25**
   - Denoise: **Median Filter**
   - Kernel: **3**
4. Do not calibrate the scale, select an ROI, or manually edit the mask.
5. Wait until image processing finishes and check the **Binary** and **Result** views.
6. Run the following analyses:
   - **Area Ratio**
   - **Skeleton Length**
   - **Average Width**
   - **Crack Segmentation**
   - **Junction Angles**
   - **Soil Clod Analysis**
7. In the **Crack Segmentation** window, click **Save Image** to save the segmentation result.
8. In the **Junction Angle Analysis** window, click crack junctions to display their angles, then click **Save Image**.
9. In the **Soil Clod Analysis** window, click **Save Image**.
10. Open the **Export** menu and select **Export Excel**.

Compare the reproduced outputs with:

- `Examples/Analysis_Result_5.xlsx`
- `Examples/Crack Segmentation-5.png`
- `Examples/Junction Angle Analysis-5.png`
- `Examples/Soil Cloud Analysis-5.png`

The main values in `Examples/Analysis_Result_5.xlsx` are:

| Metric | Reference value |
|---|---:|
| Area Ratio | 5.0002% |
| Total Crack Length | 4955.80 px |
| Average Width | 9.47 px |
| Maximum Width | 21.18 px |

The reference workbook uses the earlier label `Sauvola (Recommended)`. Select `Sauvola` in the current interface; the processing method is the same.
