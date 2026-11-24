# SEE Final Submission – Team 4

Scientific Experimentation and Evaluation, Hochschule Bonn-Rhein-Sieg.
Team: Shivank Sharma, Ishank Sharma, Vanpreet Kaur Chopra.

**Report:** [`SEE_Sharma_04_FinalReport.pdf`](SEE_Sharma_04_FinalReport.pdf). It contains all six
assignments, and each part ends with the steps needed to repeat it.

| Folder | Assignment | Contents |
|---|---|---|
| `report/` | A1 Design of the LEGO EV3 experiment | error propagation code, expected poses, robot photos |
| `report_hw02/` | A2 Running the LEGO experiment | pencil readings, brick (encoder) files, end poses, combined class data, photos, videos |
| `report_hw03/` | A3 Statistics of the LEGO data | analysis code |
| `report_hw04/` | A4 Camera calibration | calibration images, square-size readings, results, code, checkerboard |
| `report_hw05/` | A5 KUKA youBot experiment | our OptiTrack takes and ROS bags, exported poses, combined class data, photos, videos |
| `report_hw06/` | A6 Statistics of the youBot data | analysis code |

## Running the code
Python 3 with NumPy, SciPy, Matplotlib and OpenCV (A4); A5 also needs `pip3 install rosbags`.
Run each script from its assignment folder:

```
cd report_hw03 && python3 code/statistics.py        # reads report_hw02/data/combined_*.csv
cd report_hw06 && python3 code/evaluate_youbot.py   # reads report_hw05/data/combined_*.csv
cd report_hw04 && python3 code/calibrate.py
```

`report_hw02/code/process_runs.py` and `report_hw05/code/process_youbot.py` also read the raw
files of the other teams. Those files are not published here; the combined class data they
produce are included.

The provided course programs (`EV3_Control.py`, the youBot control scripts) belong to the
course and are not included.
