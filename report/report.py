"""Build the A4 B.Tech implementation report from checked experiment artifacts."""

from collections import Counter
import argparse
import csv
import gzip
import html
from io import BytesIO
import base64
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents


HERE = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=HERE / 'drone' / 'results')
parser.add_argument('--replays', type=Path, default=HERE / 'replays.json')
parser.add_argument('--out', type=Path, default=HERE.parent / 'outputs' / 'drone-project-report.pdf')
parser.add_argument('--fonts', type=Path, default=Path('C:/Windows/Fonts'))
parser.add_argument('--details', type=Path, default=HERE / 'report-details.json')
parser.add_argument('--gazebo-result', type=Path)
parser.add_argument('--gazebo-image', type=Path)
args = parser.parse_args()
WORK, OUTPUT = args.results, args.out
details = json.loads(args.details.read_text(encoding='utf-8')) if args.details.exists() else {}
gazebo = json.loads(args.gazebo_result.read_text(encoding='utf-8')) if args.gazebo_result else None
gazebo_abstract = (
    f'The recorded Gazebo Fortress demonstration used ROS 2 and MiDaS on Grove seed {gazebo["seed"]} '
    f'and reached the goal after {gazebo["elapsed_sim_s"]:.2f} simulated seconds.'
    if gazebo else 'The ROS 2 and Gazebo integration provides a separate Linux demonstration path.'
)
for label, file in [('Times', 'times.ttf'), ('Times-Bold-TTF', 'timesbd.ttf'),
                    ('Times-Italic-TTF', 'timesi.ttf'), ('Times-Bold-Italic', 'timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(label, str(args.fonts / file)))
pdfmetrics.registerFontFamily('Times', normal='Times', bold='Times-Bold-TTF',
                             italic='Times-Italic-TTF', boldItalic='Times-Bold-Italic')
INK = colors.HexColor('#18252b')
MUTED = colors.HexColor('#445b63')
FAINT = colors.HexColor('#e3ebea')
ACCENT = colors.HexColor('#1b685a')
WIDTH, HEIGHT = A4
W = WIDTH - 144
story = []


def load(name, default=None):
    path = WORK / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def read_csv(name):
    path = WORK / name
    with (gzip.open(path, 'rt', encoding='utf-8', newline='') if path.suffix == '.gz' else
          path.open('r', encoding='utf-8', newline='')) as source:
        return list(csv.DictReader(source))


result = load('summary.json')
config = load('config.json')
study = load('study.json')
drift = load('drift_summary.json')
data = load('rgbd_data_card.json')
nyu = (load('nyu_summary.json', []) or [None])[0]
kitti = load('kitti_summary.json')[0]
depth = load('depth_summary.json')
profiles = [load(f'profile_{key}.json') for key in ('onnx', 'torch', 'tensorrt')]
nyu_affine = (load('nyu_alignment_diagnostic.json', []) or [None])[-1]
tracking = load('mlflow_run_ids.json')
runs = read_csv('courses.csv')
trials = read_csv('optuna_trials.csv')
export = read_csv('rgbd_descriptors.csv')
replays = json.loads(args.replays.read_text(encoding='utf-8'))
mid, oracle, base = (result[key] for key in ('midas', 'oracle', 'baseline'))
camera = config['camera']
nyu_short = (f"{nyu['images']} NYU Depth V2 frames (δ₁={nyu['delta1']:.3f})" if nyu else
             "the NYU Depth V2 test protocol, for which results are pending")


def style(name, face='Times', size=12, leading=18, **kwargs):
    return ParagraphStyle(name, fontName=face, fontSize=size, leading=leading,
                          textColor=INK, **kwargs)


BODY = style('Body', alignment=TA_JUSTIFY, spaceAfter=8, allowOrphans=0, allowWidows=0)
H1 = style('Chapter', 'Times-Bold-TTF', 16, 23, spaceAfter=13, keepWithNext=True)
H2 = style('Section', 'Times-Bold-TTF', 14, 21, spaceBefore=10, spaceAfter=5, keepWithNext=True)
TITLE = style('Title', 'Times-Bold-TTF', 24, 31, alignment=TA_CENTER, spaceAfter=24)
CENTER = style('Centre', alignment=TA_CENTER, spaceAfter=10)
LEAD = style('Caption', 'Times-Italic-TTF', 11, 15, spaceBefore=6, spaceAfter=7)
CELL = style('Cell', size=11, leading=14)
CELL_HEAD = style('CellHead', 'Times-Bold-TTF', 11, 14, textColorOverride=None)
NOTE = style('Note', 'Times-Italic-TTF', 11, 16, spaceAfter=7)
EQ = style('Equation', 'Times-Italic-TTF', 12, 19, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8)
REF = style('Reference', size=10.5, leading=14, spaceAfter=7, alignment=TA_LEFT)
FORM = style('FormLabel', alignment=TA_LEFT, spaceAfter=11)


def clean(s):
    result = html.escape(str(s))
    for glyph, sub in {'₁':'1','₂':'2','₃':'3','ₖ':'k','ᵢ':'i','ⱼ':'j','ₓ':'x','ᵧ':'y'}.items():
        result = result.replace(glyph, f'<sub>{sub}</sub>')
    return result


def paragraph(s, st=BODY, markup=False):
    return Paragraph(s if markup else clean(s), st)


def p(*items):
    story.extend(paragraph(item) for item in items)


def centered(s):
    story.append(paragraph(s, CENTER))


def gap(n=12):
    story.append(Spacer(1, n))


def page():
    story.append(PageBreak())


def front(s):
    story.append(paragraph(s, H1))


def h(s):
    story.append(paragraph(s, H2))


def chapter(n, s):
    page()
    marker = paragraph(f'{n}. {s}', H1)
    marker.index = ('TOCEntry', 0, f'{n}. {s}', f'chapter-{n}')
    story.append(marker)


table_number = figure_number = 0


def table(caption, headers, rows, fractions=None):
    global table_number
    table_number += 1
    label = f'Table {table_number}. {caption}'
    cap = paragraph(label, LEAD)
    cap.keepWithNext = True
    cap.index = ('TableEntry', 0, label, f'table-{table_number}')
    data_rows = [[paragraph(c, CELL_HEAD if i == 0 else CELL) for c in row]
                 for i, row in enumerate([headers, *rows])]
    widths = [W * f for f in fractions] if fractions else None
    block = Table(data_rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    block.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), FAINT),
        ('LINEBELOW', (0, 0), (-1, 0), .65, MUTED),
        ('LINEBELOW', (0, 1), (-1, -1), .25, FAINT),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    story.extend([cap, block, Spacer(1, 7)])


def equation(num, s):
    story.append(paragraph(f'{s}     ({num})', EQ))


def figure(caption, graphic):
    global figure_number
    figure_number += 1
    label = f'Figure {figure_number}. {caption}'
    cap = paragraph(label, LEAD)
    cap.index = ('FigureEntry', 0, label, f'figure-{figure_number}')
    story.append(KeepTogether([graphic, cap]))


def diagram(items):
    height = 64 if len(items) <= 4 else 120
    drawing = Drawing(W, height)
    columns = 4 if len(items) >= 4 else len(items)
    cols_w = W / columns
    for i, item in enumerate(items):
        x, y = (i % columns) * cols_w + 3, height - 46 - (i // columns) * 58
        drawing.add(Rect(x, y, cols_w - 13, 39, fillColor=FAINT, strokeColor=None))
        for j, line in enumerate(item.split('\n')):
            drawing.add(String(x + (cols_w - 13) / 2, y + 24 - j * 13, line, textAnchor='middle',
                               fontName='Times', fontSize=10, fillColor=INK))
        if i % columns < columns - 1 and i + 1 < len(items):
            drawing.add(Line(x + cols_w - 13, y + 20, x + cols_w - 1, y + 20, strokeColor=ACCENT))
    return drawing


def bar_chart(labels, values, limit, fmt):
    d = Drawing(W, 28 + len(labels) * 36)
    y0 = d.height - 32
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = y0 - idx * 36
        d.add(String(2, y + 4, label, fontName='Times-Bold-TTF', fontSize=11, fillColor=INK))
        d.add(Rect(124, y, W - 186, 14, fillColor=FAINT, strokeColor=None))
        d.add(Rect(124, y, (W - 186) * value / limit, 14, fillColor=ACCENT, strokeColor=None))
        d.add(String(W - 55, y + 3, fmt(value), fontName='Times', fontSize=11, fillColor=INK))
    return d


def histogram(a, b, label):
    d = Drawing(W, 135)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ab, bins = np.histogram(a, np.linspace(lo, hi, 16))
    bb, _ = np.histogram(b, bins)
    peak = max(ab.max(), bb.max())
    plot_w, cell_w = W - 65, (W - 65) / len(ab)
    for n, (left, right) in enumerate(zip(ab, bb)):
        for i, height in enumerate((left, right)):
            d.add(Rect(33 + cell_w * n + i * cell_w / 2, 24, cell_w / 2 - 1,
                       78 * height / peak, fillColor=ACCENT if i == 0 else colors.HexColor('#bd7c3e'),
                       strokeColor=None))
    d.add(Line(33, 23, 33+plot_w, 23, strokeColor=MUTED))
    for x, text in [(33, f'{lo:.2f}'), (33+plot_w, f'{hi:.2f}'), (W/2, label)]:
        d.add(String(x, 10, text, textAnchor='middle', fontName='Times', fontSize=10))
    d.add(String(33, 116, 'Corridor', fontName='Times-Bold-TTF', fontSize=10, fillColor=ACCENT))
    d.add(String(90, 116, 'Grove', fontName='Times-Bold-TTF', fontSize=10,
                 fillColor=colors.HexColor('#9c5d27')))
    return d


def corr_chart(keys, matrix):
    d = Drawing(W, 187)
    length = 30
    labels = ['Bright', 'Contrast', 'Valid', 'Near', 'Median']
    for i, label in enumerate(labels):
        d.add(String(136, 145 - i*length, label, textAnchor='end', fontName='Times', fontSize=10))
        d.add(String(154 + i*length, 170, label, fontName='Times', fontSize=8))
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            rgb = (0.77-.60*abs(value), .86-.31*abs(value), .85-.38*abs(value)) if value >= 0 else (
                .94-.08*abs(value), .87-.40*abs(value), .75-.43*abs(value))
            x, y = 142 + j*length, 136 - i*length
            d.add(Rect(x, y, 28, 27, fillColor=colors.Color(*rgb), strokeColor=None))
            d.add(String(x+14, y+10, f'{value:.2f}', textAnchor='middle', fontName='Times', fontSize=9,
                         fillColor=colors.white if value > .73 else INK))
    return d


def replay_figure(run, i):
    f = run['frames'][i]
    images = [PILImage.open(BytesIO(base64.b64decode(f[key].split(',', 1)[1])))
              for key in ('rgb', 'depth')]
    panel = PILImage.new('RGB', (260, 80), (24, 37, 43))
    panel.paste(images[0], (0,0))
    panel.paste(images[1], (132,0))
    img = BytesIO()
    panel.resize((1040,320), PILImage.Resampling.LANCZOS).save(img, format='PNG')
    img.seek(0)
    return Image(img, width=W, height=W*80/260)


def index(event, font=11, leading=16):
    idx = TableOfContents(levelStyles=[style('IndexRow'+event, size=font, leading=leading)],
                          dotsMinLevel=0)
    # ReportLab's default listener filters the topic to TOCEntry.
    idx.notify = lambda kind, value, listener=idx: listener.addEntry(*value) if kind == event else None
    story.append(idx)


def roman(n):
    labels = ['', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix']
    return labels[n]


def footer(canvas, doc):
    pn = doc.page
    canvas.setFont('Times', 11)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(WIDTH/2, 46, roman(pn) if pn <= 9 else str(pn - 9))


class Report(BaseDocTemplate):
    def afterFlowable(self, flowable):
        entry = getattr(flowable, 'index', None)
        if entry:
            kind, level, label, ident = entry
            self.canv.bookmarkPage(ident)
            if kind == 'TOCEntry':
                self.canv.addOutlineEntry(label, ident)
            self.notify(kind, (level, label, self.page - 9, ident))


gap(37)
centered(details.get('institution', '[INSTITUTION NAME]'))
centered(details.get('department', '[DEPARTMENT NAME]'))
gap(28)
centered('THIRD YEAR B.TECH. IN ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING')
centered(f'Project Life Cycle Management • {details.get("course_code", "[Course code]")}')
gap(38)
story.append(paragraph('Autonomous Drone Obstacle Avoidance Using Monocular Depth Estimation', TITLE))
centered('Project 6 • Final Project Implementation Report')
gap(28)
centered(f'Group {details.get("group", "[number]")}   |   Academic year {details.get("academic_year", "[year]")}')
students = details.get('students', [
    {'name': f'[Student {i} name]', 'roll_number': '[Roll number]'} for i in range(1, 5)])
for student in students:
    centered(f'{student["name"]}   {student["roll_number"]}')
gap(12)
centered(f'Faculty mentor   {details.get("mentor", "[Name]")}')
page()
front('Certificate of Project Completion')
p('This page is reserved for the institution’s completion certificate. The faculty mentor and department should confirm the scope of the work accepted for assessment and record the applicable submission status.')
gap(36)
for item in ['Institution and department', 'Project group and academic year',
             'Approved wording or reference number', 'Faculty mentor signature and date',
             'Department approval and date']:
    story.append(paragraph(item + ' ______________________________', FORM))
page()
front('Declaration of Originality')
p('The submitting group should complete this declaration with its institution’s approved wording. Each member should verify the contribution matrix, acknowledge external sources and assistance in accordance with institutional policy, and sign only the final version they have reviewed.')
gap(30)
for item in range(1, 5):
    story.append(paragraph(f'Name and signature of student {item} ______________________________', FORM))
gap(16)
p('Place ___________________________    Date ___________________________')
page()
front('Acknowledgement')
p('The project uses the published MiDaS v2.1 Small model and draws on the NYU Depth V2 and KITTI research datasets. The maintainers of ROS 2, Gazebo, ONNX Runtime, PyTorch, Optuna, MLflow and Evidently provide the open tools used or evaluated in the implementation.')
p('The submission group may add names and specific contributions from faculty, laboratory staff and peers after verifying each acknowledgement.')
page()
front('Abstract')
p(
    'A lightweight drone can use a single RGB camera to estimate the space ahead, but monocular relative depth alone does not provide a reliable metric stopping distance. This project implements a fixed-altitude autonomous drone simulation with a perspective camera, seeded corridor and grove courses, and a reactive controller with a forward-motion override. MiDaS v2.1 Small predicts relative inverse depth. A calibration fitted only to development scenes converts its output to approximate metres for control. Matching ground-truth-depth and straight-line runs isolate the effect of perception.',
    f'The local dataset contains {data["pairs"]:,} generated RGB-depth pairs. Offline depth evaluation includes 250 held-out synthetic pairs, 25 KITTI Eigen-list frames from one drive, and {nyu_short}. A six-trial Optuna study tuned the controller. ONNX Runtime CPU was used for the frozen evaluation, with its results recorded in MLflow.',
    f'In the local ray-based simulator, MiDaS reached the goal without collision in {mid["goals"]} of 50 courses (92.0%, 95% Wilson interval 81.2%–96.8%). Ground-truth depth reached 50 of 50 and the depth-free baseline reached zero. In the dedicated stop-only tests, the MiDaS arm collided in {mid["safety"]["collided"]} of 20 cases. ONNX Runtime achieved a {mid["latency"]["loop_ms"]["p95"]:.1f} ms 95th-percentile loop time in the local simulator. A fog injection triggered the degraded-perception rule. {gazebo_abstract} The safety result and simplified dynamics limit the prototype to simulated use.'
)
page()
front('Table of Contents')
index('TOCEntry')
page()
front('List of Figures')
index('FigureEntry')
page()
front('List of Tables')
index('TableEntry', font=10.5, leading=12)
page()
front('List of Abbreviations')
for short, long in [
    ('AbsRel', 'Absolute relative depth error'), ('AI', 'Artificial intelligence'),
    ('CI/CD', 'Continuous integration and delivery'),
    ('CRISP-DM', 'Cross-Industry Standard Process for Data Mining'),
    ('DGCA', 'Directorate General of Civil Aviation, India'),
    ('EDA', 'Exploratory data analysis'), ('FP16 / FP32', '16-bit / 32-bit floating point'),
    ('FPS', 'Frames per second'), ('IQR', 'Interquartile range'),
    ('ML', 'Machine learning'), ('NFR', 'Non-functional requirement'),
    ('ONNX', 'Open Neural Network Exchange'), ('PSI', 'Population Stability Index'),
    ('RGB-D', 'Red, green, blue and aligned depth'), ('RMSE', 'Root mean squared error'),
    ('ROS 2', 'Robot Operating System 2'), ('SDF', 'Simulation Description Format'),
    ('TDSP', 'Team Data Science Process'), ('UAV', 'Unmanned aerial vehicle')]:
    centered(f'{short}  •  {long}')


chapter(1, 'Project Introduction and Problem Definition')
h('1.1 Background and problem statement')
p('Small drones used for inspection may have limited room for sensing hardware. A forward-facing RGB camera offers a compact sensing path. The engineering problem is to infer usable free space from one image and choose a velocity command before the drone progresses too far. Apparent obstacle size, poor texture and ambiguous image scale make the lowest estimated frontal depth an imperfect safety signal.',
  'The implementation tests whether a static course can be crossed using estimated depth, and then measures the degradation against an oracle using identical starts, goals, geometry and controller settings. A separate stop-only test exposes faults that normal steering can hide.')
h('1.2 Objectives and scope')
table('Project objectives and acceptance evidence',
      ['Objective', 'Acceptance evidence'], [
          ('Reactive navigation', 'At least 45 collision-free arrivals in 50 unseen courses using estimated depth. The project brief specifies Gazebo.'),
          ('Real-time perception', 'At least 10 depth frames/s and a 100 ms camera-to-command budget at 2 m/s.'),
          ('Depth accuracy', 'Report δ₁, δ₂, AbsRel and RMSE on NYU, KITTI and held-out simulation images. NYU δ₁ target is 0.80.'),
          ('Stop override', 'Zero collisions in 20 dedicated frontal tests with steering disabled.'),
          ('Lifecycle evidence', 'Seeded datasets and runs, tuning history, logging, drift exercise, CI configuration and safety review.')
      ], [.27,.73])
p('The courses contain static obstacles at fixed altitude. The policy has access to pose and a goal bearing, with obstacle information supplied by the selected depth source. Flight outside simulation, moving obstacles, SLAM, global planning and model training are outside this implementation.')
h('1.3 Proposed solution')
p('The RGB frame passes through a small pretrained inverse-depth network. The controller evaluates a fan of candidate headings, steers toward reachable space and applies the forward stop before sending a command. Independent geometry samples the swept path, so the controller cannot mark its own run as collision-free.')
figure('Perception and control stages.', diagram(['Perspective\nRGB camera','MiDaS-small\ninverse depth','Calibration and\nreactive control','Velocity command\nand telemetry']))
h('1.4 Course and vehicle definitions')
table('Course, camera and endpoint definitions', ['Item', 'Definition in the measured simulator'], [
    ('Course assignment', 'Even seeds produce corridors. Odd seeds produce groves. Every test includes three procedural foreground obstacles.'),
    ('Goal and tolerance', 'The vehicle starts at (0, 0), flies at 1.25 m, and succeeds within 0.55 m of (16, 0) after the swept clearance check.'),
    ('Vehicle footprint', 'A disc of radius 0.28 m defines the collision envelope at the nominal camera height.'),
    ('Mission time limit', '19 simulated seconds. The dedicated stop-only run watches for contact or an unintended restart over 8 seconds.'),
    ('Success counts', 'The experiment keeps goal, collision and timeout as distinct terminal conditions. A stop-only halt is recorded separately.')
], [.31,.69])
p('The start and end positions are fixed while obstacle geometry changes with the seed. This means a baseline can receive the same destination as the avoidance arms without receiving a map. The finite test time penalises a controller that becomes trapped or repeatedly stops.')

chapter(2, 'Project Life Cycle Frameworks')
h('2.1 AI/ML development stages')
p('Requirements fixed the collision and latency targets before model integration. A deterministic renderer supplied paired RGB-D for data quality work. Calibration and controller tuning then used distinct development seeds. Validation was checked before the frozen evaluation. The failure review compared oracle depth and model depth, and monitoring exercised a simulated fog shift.')
figure('Lifecycle stages and the review cycle.', diagram(['Scope and\nrequirements','Collect and\nassess data','Calibrate and\ntune policy','Freeze and\nevaluate','Profile and\npackage','Monitor\nscene drift','Review\nfailures','Revise with\nnew seeds']))
h('2.2 Framework comparison')
table('Lifecycle approaches and use in this project', ['Framework', 'Emphasis', 'Use here'], [
    ('CRISP-DM [7]', 'Iterative investigation of data and objectives.', 'Primary structure. It connects sensor errors to evaluation and feasibility.'),
    ('Microsoft TDSP [8]', 'Team responsibilities and delivery stages.', 'Contribution matrix, repository structure and handoff checklist.'),
    ('Google ML lifecycle [9]', 'Operational pipelines, validation and continuous delivery.', 'Runtime monitoring and quality-gate design.')
], [.25,.32,.43])
p('Conventional software tests can assert exact outputs for a fixed seed or a missing frame. Model behaviour also depends on the distribution of scene appearances, so statistical tests examine held-out courses and uncertainty. The required maintenance includes checking whether perception still supports the original decision under drift.')
h('2.3 CRISP-DM phase mapping')
table('CRISP-DM mapping', ['Phase', 'Project evidence'], [
    ('Business understanding', 'Simulation-only scope and quantitative safety, perception and deadline requirements.'),
    ('Data understanding', 'Camera intrinsics, field-of-view coverage, invalid depth and brightness distributions.'),
    ('Data preparation', 'Aligned RGB-D pairs, split by course seed, reference depth masks and per-image metric scaling for offline comparison.'),
    ('Modelling', 'MiDaS inference, development-only calibration and an Optuna study of the reactive controller.'),
    ('Evaluation', 'Matched three-arm ablation, domain benchmarks, timed inference and stop-only cases.'),
    ('Deployment', 'Portable runner, browser playback and Linux ROS 2/Gazebo configuration.')
], [.30,.70])
p('The ML project includes problem framing, policy choices, failure review and governance. The operational pipeline is the repeatable sequence that accepts RGB and emits velocity commands. Assessment considers both.')

chapter(3, 'Feasibility and Requirements Engineering')
h('3.1 Four-dimensional feasibility')
table('Feasibility matrix', ['Dimension', 'Finding', 'Decision'], [
    ('Technical', 'The CPU prototype estimates depth in time on the measured Windows machine. The safety arm failed. The Gazebo batch acceptance test remains open.', 'Conditional'),
    ('Economic', 'No aircraft, sensors or paid compute were used for the prototype. The existing computer and its electricity have a cost outside the project cash estimate.', 'Feasible for the simulation scope'),
    ('Operational', 'Seeded CLI, replay and manifests support demonstration and maintenance. Production operations would require independent safety validation.', 'Prototype use'),
    ('Ethical', 'Synthetic scenes contain no people. External RGB imagery is held outside shared repositories. The failure evidence argues against deployment on aircraft.', 'Simulation only')
], [.22,.53,.25])
h('3.2 Deadline and stopping envelope')
p('The design allows at most 0.20 m of travel while a new frame becomes a command at the maximum cruise speed. In (1), L is processing latency, d is the permitted travel during that latency, and v is the commanded cruise speed.')
equation(1, 'L = d / v = 0.20 m / (2.0 m/s) = 0.10 s')
p('This deadline covers reaction distance. A flight stopping envelope would also need braking distance, actuation lag and measurement uncertainty. For illustration, a constant 2 m/s² deceleration from 2 m/s adds 1.00 m before the vehicle stops. The local simulator applies the command at each 0.1 s step and does not model that braking term.')
h('3.3 Specification and traceability')
table('Functional, data and ethical acceptance criteria', ['ID', 'Criterion', 'Evidence'], [
    ('FR1–FR2', 'Acquire RGB and generate dense depth, then produce yaw rate and forward velocity.', 'Inference backend and per-frame CSV. ROS topics are authored.'),
    ('FR3–FR4', 'Override forward motion at the threshold. Reach the goal autonomously.', 'Stop traces and goal/collision outcomes.'),
    ('FR5–FR6', 'Log depth descriptors and clearance. Reproduce layouts from seeds.', 'Compressed frame CSV, tests and seed registry.'),
    ('DATA1–2', 'Use published datasets and generate at least 5,000 paired simulator frames with camera metadata.', 'Benchmark output and manifest. Gazebo capture awaits execution.'),
    ('ETH1–3', 'Record hazards, privacy implications and honest estimated-depth results.', 'Chapters 7, 14 and 15.')
], [.19,.41,.40])
p('Feasibility decision. Proceed with simulation coursework and Linux verification. Explicitly gate any move beyond this prototype on a demonstrated safe sensor-and-control system, with independent validation of perception and braking.')

chapter(4, 'Stakeholder Identification and Management')
h('4.1 Roles and potential impact')
figure('Stakeholder map for the simulation project.', diagram(['Course faculty\nand assessors','Development\nteam','Inspection\noperators','Data stewards','Bystanders and\ncommunities','Property\nowners','Airspace\nregulators','Insurers and\nsafety reviewers']))
table('Influence and interest matrix', ['Stakeholder', 'Influence / interest', 'Engagement and evidence'], [
    ('Faculty and project team', 'High / high', 'Review acceptance criteria, demonstrations, versioned evidence and contribution records.'),
    ('Operators and property owners', 'High / high in use', 'Review understandable stop behaviour, mission scope and abort procedures.'),
    ('Airspace regulators and insurers', 'High / context dependent', 'Future review of safety classification, operator responsibilities and risk evidence.'),
    ('Bystanders and data subjects', 'Variable / high', 'Prioritise no capture where possible. Assess notice, recording controls and complaint channels for any future trials.'),
    ('Dataset providers', 'Medium / medium', 'Preserve attribution and distribution terms.'),
    ('Potential adversary', 'Variable / misuse interest', 'Track foreseeable spoofing and privacy-invasive use. Restrict the intended use to simulated civil inspection.')
], [.28,.25,.47])
h('4.2 Conflict management')
p('A policy can complete a course despite missing the obstacle in the depth map, which complicates discussions about accuracy and interpretability. The review therefore includes actual command inputs, clearance and an independent stop-only test. A single-camera payload objective does not override bystander safety.',
  'Improving coverage requires representative imagery, while retention creates privacy exposure. The prototype generates its own camera frames for course development and stores only metrics and manifests from external datasets. The project team can use recorded simulation traces for feedback without collecting new imagery of people.',
  'Schedule pressure can obscure unresolved acceptance conditions. Demonstration results, draft integrations and pending measurements each have explicit statuses so the coordinator can decide what belongs in the assessed submission.')
h('4.3 Review schedule and disputed decisions')
table('Evidence to bring to stakeholder reviews', ['Review', 'Participants and input', 'Decision'], [
    ('Requirements', 'Team and mentor compare intended mission, deadline and collision targets.', 'Approve seed split and test definitions before tuning.'),
    ('Data and ethics', 'Data steward checks acquisition terms and retention, with bystander concerns represented in the risk record.', 'Release metrics, source and synthetic outputs.'),
    ('Failure review', 'Controls and perception leads replay joint depth, commands and clearance.', 'Identify stop failures that steering conceals.'),
    ('Handoff', 'Faculty and team compare the report with the portable batch and cloud Gazebo evidence.', 'Sign only institution-approved scope and contributions.')
], [.18,.44,.38])

chapter(5, 'Data Acquisition, Governance and Versioning')
h('5.1 Sources and formats')
table('Dataset inventory', ['Source', 'Observations and features', 'Depth labels and split'], [
    ('Ray-cast RGB-D v1', f'{data["pairs"]:,} frames at 128 × 80. Three RGB channels and one aligned depth channel. {data["total_shard_bytes"]/1e6:.1f} MB of compressed shards.', 'Axial millimetres. Export seeds 400–419, separate from frozen mission tests.'),
    ('NYU Depth V2 [2]', 'Indoor 640 × 480 Kinect RGB-D. Official labelled release has 1,449 frames.', f'Official 654-frame split. {str(nyu["images"]) + " evaluated." if nyu else "Evaluation pending here."}'),
    ('KITTI raw [3], [4]', '25 paired file IDs from one 2011-09-26 drive in the Eigen list. Three RGB channels plus sparse projected points.', 'Rectified axial depth in metres, reprojection and nearest-return merge.'),
    ('Holdout ray-cast', '250 paired frames from five poses on each frozen course.', '20 m far sentinel excluded. Used for metrics after tuning.')
], [.23,.40,.37])
figure('Lineage of local synthetic RGB-depth data.', diagram(['Versioned seed\nand camera pose','Geometry and\nRGB-D renderer','Aligned arrays\nand manifest','Depth metrics\nand EDA']))
h('5.2 Provenance and retention')
p('The renderer generates RGB and depth from the same rays, and the synthetic frames contain no human subjects. The NYU dataset documents Kinect capture and filled depth labels [2]. KITTI provides synchronised cameras, Velodyne scans and calibration files [3]. Reported KITTI metrics are a local research comparison on a small subset, and the benchmark server has its own submission policy.',
  f'The synthetic camera fₓ and fᵧ equal {camera["fx"]:.2f} px, with principal point ({camera["cx"]:.1f}, {camera["cy"]:.1f}). The HFOV is 88° and the camera height is 1.25 m. KITTI uses the supplied date-specific rectified projection and extrinsics. NYU depth is paired with the released RGB image, using the dataset’s pre-alignment and crop without recalibrating Kinect.')
h('5.3 Version history and reproducibility')
p('The baseline used the default policy, the Optuna run selected a parameter set on seeds 200–203, and release 1.0 applied that frozen set to seeds 1000–1049. The manifest records pose, source seed, frame content hash and shard index. Shard hashes record the exported byte-level version. The project packages Python sources and a GitHub workflow, ready for a repository selected by the group.')
h('5.4 Benchmark identity and transforms')
table('Benchmark lineage and identity controls', ['Source stage', 'Recorded action', 'Validation condition'], [
    ('NYU test IDs', 'Load testNdxs from the standard 654-frame split MAT and convert to zero-based HDF5 offsets.', 'The loader requires 654 distinct integers.'),
    ('KITTI raw pair', 'Read camera PNG and the corresponding Velodyne scan; project with the date-specific calibration.', 'The matched triplet is drive, frame number and camera side.'),
    ('Crop and depth gate', 'Central NYU crop, Eigen crop for KITTI, finite positive distances up to dataset-specific caps.', 'Every benchmark record includes its number of valid pixels.'),
    ('Registry output', 'Write image metrics, source digests and aligned aggregate values.', 'A rerun can compare source digests before comparing scores.')
], [.22,.44,.34])

chapter(6, 'Data Quality Assessment and Exploratory Data Analysis')
h('6.1 Quality dimensions')
keys = ['brightness','contrast','valid_fraction','near_fraction','median_depth_m']
labels = ['Brightness','Contrast','Valid depth fraction','Near fraction','Median depth (m)']
summary = []
for key, label in zip(keys,labels):
    vec = np.asarray([float(x[key]) for x in export if x[key]])
    q1, q3 = np.percentile(vec,[25,75])
    iqr = q3-q1
    outliers = np.sum((vec < q1-1.5*iqr) | (vec > q3+1.5*iqr))
    summary.append((label, f'{vec.mean():.3f}', f'{np.median(vec):.3f}', f'{vec.var(ddof=1):.4f}',
                    f'{vec.std(ddof=1):.3f}', f'{vec.min():.3f}–{vec.max():.3f}',
                    f'{q1:.3f}–{q3:.3f}', f'{iqr:.3f}', str(outliers)))
table('Descriptive statistics on 5,000 generated frames', 
      ['Feature','Mean','Median','Var.','SD','Min–max','Q1–Q3','IQR','Flag'], summary,
      [.19,.09,.10,.10,.08,.13,.13,.09,.09])
p(f'Completeness is checked at the RGB-D pair and pixel levels. The manifest contains {data["pairs"]:,} pairs, of which {data["unique_content_hashes"]:,} have unique content hashes. A far-clip pixel is a censored measurement: sky or unobstructed distance may exceed 20 m. Its missing true range depends on scene geometry, so missing completely at random is implausible. Files have uniform 128 × 80 geometry and shared clocks in the local renderer. Synthetic collection time is represented by the seed and pose lineage.')
group_a, group_b = (np.array([float(row['valid_fraction']) for row in export if row['scene'] == scene])
                    for scene in ('Corridor','Grove'))
figure('Distribution of the valid depth fraction across frames.', histogram(group_a,group_b,'Valid depth fraction'))
h('6.2 Outliers and correlation')
equation(2, 'IQR = Q₃ − Q₁. Flag x < Q₁ − 1.5 IQR or x > Q₃ + 1.5 IQR.')
p('Q₁ and Q₃ are the 25th and 75th percentiles of each feature across exported frames. NumPy quantiles produce the flags in the descriptive statistics table. Retaining the geometry-driven tails preserves contrast and near-field conditions relevant to navigation.')
matrix = np.array([[float(row[key]) for key in keys] for row in export if all(row[k] for k in keys)])
figure('Pearson correlation of frame descriptors.', corr_chart(keys,np.corrcoef(matrix,rowvar=False)))
table('Quality issues and corrective action', ['Observed issue', 'Treatment'], [
    ('Far clip and sky pixels', 'Exclude 20 m sentinel from valid-pixel depth evaluation. Keep the valid fraction for EDA.'),
    ('Sparse or duplicate KITTI projections', 'Discard invalid cells, keep the nearest projected return per pixel and store valid counts.'),
    ('Different rendering and camera geometry', 'Record camera model and scene, keep test separated by seed, and report domain-specific results.'),
    ('Ambiguous monocular scale', 'Fit the live transform on development pairs, then compare against per-image label-aligned metrics offline.')
], [.37,.63])

chapter(7, 'Data Ethics and Bias Assessment')
h('7.1 Privacy and data governance')
p('The generated courses consist of floors, walls, boxes and simplified trunks. They include no people or identifiers. External RGB and depth inputs are acquired for benchmark comparison under the providers’ documented terms. The package includes processed aggregate metrics and file identifiers. Original benchmark imagery stays with the source downloads.',
  'For a future real deployment, GDPR principles where applicable include minimisation, purpose limitation and storage limitation [11]. A design review should establish the lawful basis and local obligations before collection. An RGB camera can reveal a person’s presence even when the model only outputs depth. On-device processing, no default frame retention and auditable access control reduce avoidable exposure.')
table('Data ethics checklist', ['Check', 'Status'], [
    ('Purpose limit', 'Simulation and offline model evaluation.'),
    ('Attribution and use terms', 'Official dataset sources and third-party split list cited.'),
    ('Recording minimisation', 'Demo contains only generated pixels. No real-flight collection.'),
    ('Affected-group review', 'Documented risks to bystanders, operators and property owners. Real-world consultation would be required before trials.'),
    ('Release review', 'Separate benchmark downloads from shareable code and evidence.')
], [.36,.64])
h('7.2 Scene bias and fairness')
normal = [r for r in runs if r['mode']=='midas' and r['safety']=='False']
by_scene=[(s, [r for r in normal if r['scene']==s]) for s in ('Corridor','Grove')]
table('Navigation results by synthetic scene', ['Scene', 'Goals', 'Collision count'], [
    (s, f'{sum(r["status"]=="goal" for r in rs)} / {len(rs)}', str(sum(r['status']=='collision' for r in rs)))
    for s,rs in by_scene
], [.45,.30,.25])
p('Scene stratification tests whether one simplified environment dominates the aggregate result. The paired design has 25 courses of each type. The geometry and textures omit dynamic parties, reflective surfaces and weather beyond synthetic fog. KITTI’s one-drive sample and NYU’s indoor data provide domain comparisons with limited environmental coverage.',
  'The task predicts spatial depth without demographic labels or individuals, so demographic parity, equalised odds and Fairlearn classifiers have no suitable input in this experiment. Evaluate adverse effects through operational contexts and hazard exposure when relevant data becomes available.')
h('7.3 Coverage limits observed in stop-only trials')
table('Estimated-depth stop-only cases by scene', ['Scene', 'Trials', 'Collisions', 'Safe stops'], [
    (s,len(rs),sum(r['status']=='collision' for r in rs),sum(r['status']=='stopped' for r in rs))
    for s,rs in [(s,[r for r in runs if r['scene']==s and r['mode']=='midas' and r['safety']=='True'])
                 for s in ('Corridor','Grove')]
], [.38,.20,.20,.22])
p('The stop-only scene results complement course completions. They capture performance disparities associated with environment geometry and appearance. The small seed counts and synthetic conditions limit claims about the source and persistence of those differences.')

chapter(8, 'Feature Engineering and Preprocessing')
h('8.1 Raw, learned and derived features')
table('Feature summary and transformations', ['Layer', 'Input', 'Output or use'], [
    ('Raw RGB', '128 × 80 × 3 uint8 in simulation', 'Bicubic 256 × 256 and channel-first float32 in [0, 1] for the released ONNX model.'),
    ('Learned depth', 'Relative inverse-depth pixels', 'Resize to camera geometry. Convert using fixed inverse-depth coefficients for live control.'),
    ('Controller scan', 'Central 46–54% height band', 'Minimum along each column. Project to candidate headings with the horizontal field of view.'),
    ('Monitoring', 'RGB and processed depth', 'Brightness, contrast, depth entropy and fraction below 2 m.')
], [.23,.33,.44])
figure('Shared preprocessing with separate offline scale alignment.', diagram(['RGB resize\nand scale','Network\ninverse depth','Resize back\nto camera','Metric transform\nand command']))
h('8.2 Numerical transformations')
cal = load('calibration.json')
equation(3, 'd̂ = 1 / clip(a q + b, 1/20, 4)')
p(f'In (3), q is model inverse-depth output and d̂ is axial distance in metres. Iterated least squares on calibration seeds 100–111 estimates a={cal["slope"]:.8f} and b={cal["intercept"]:.6f} using a central image band. The fit drops the highest 6% of residuals at each refit. Clipping keeps the range between 0.25 m and 20 m. For q=800, the estimated distance is {1/(800*cal["slope"]+cal["intercept"]):.2f} m.')
equation(4, 'd̂ᵢ = [1/max(qᵢ, ε)] × median(d) / median(1/max(q, ε))')
p('Equation (4) is used for offline benchmarking. Pixel i has ground-truth distance dᵢ. The validity mask selects comparable pixels and ε prevents division by zero. Median scaling can use the labels in an offline benchmark. Deployed control would have no such label access. The report includes both comparisons on the same held-out simulated images.')
h('8.3 Steering and stop calculation')
equation(5, 'f = d cos θ + s sin θ,    l = s cos θ − d sin θ')
p('Here d is a column’s axial depth, s is its lateral projection, θ is a candidate yaw offset, f is forward distance in that direction and l is lateral distance from the path. The policy scores free range inside a protected half-width, penalises target deviation and clips the rate command. The unsmoothed front minimum can force zero forward speed immediately while the yaw channel searches for an open direction. The dedicated stop-only test holds yaw at zero.')
h('8.4 Worked controller example')
example_d = 1/(800*cal['slope']+cal['intercept'])
center = camera['cx']
example_s = (center-52)/camera['fx']*example_d
p(f'For the q=800 estimate in (3), column 52 gives s=({center:.1f}−52)/{camera["fx"]:.2f} × {example_d:.2f}={example_s:.2f} m to the left of the centreline. It falls inside the protected half-width of {config["settings"]["half_width"]:.2f} m. If it is the closest return in that band, the current threshold of {config["settings"]["stop"]:.2f} m allows forward motion.')
table('Feature values before and after transformation in the example', ['Quantity', 'Before', 'After'], [
    ('Depth at selected pixel', 'Relative inverse depth 800', f'{example_d:.2f} m axial distance'),
    ('Column coordinate', '52 px', f'{example_s:.2f} m lateral projection'),
    ('Safety decision', 'Closest frontal estimate', 'Continue with speed controlled by the selected heading')
], [.31,.31,.38])

chapter(9, 'Model Selection and Baseline Development')
h('9.1 Problem type and candidates')
p('Depth estimation is dense regression on ordinal and approximate metric geometry. Collision-free goal attainment is a closed-loop outcome, so depth error is not the same target as a course completion rate. The project chose a compact pretrained encoder for CPU inference. MiDaS’s larger DPT family is a candidate when computational resources and calibration justify it [1].')
table('Candidate and baseline selection', ['Option', 'Selection basis', 'Evidence'], [
    ('MiDaS v2.1 Small', 'Relative inverse depth from RGB with a 256 × 256 small backbone.', 'Deployed in the local controller and evaluated.'),
    ('DPT Hybrid', 'Transformer-based MiDaS alternative with larger input. Needs resource and parity testing.', 'Documented alternative. No measured comparison in this release.'),
    ('Ground-truth geometry', 'Simulator supplies perfect axial measurement.', 'Oracle ablation. Measures headroom in the reactive policy.'),
    ('Straight-line policy', 'Uses goal error and forward speed with no depth.', 'Explicit zero-depth baseline. Same seed and initial pose.')
], [.28,.38,.34])
h('9.2 Training and initial policy')
p('The release reuses the published checkpoint. There is no gradient training. Project-level development consists of fitting the inverse-depth calibration and choosing policy settings. The baseline default on validation reached four of six goals. The alternative settings introduced wider candidate clearance and stronger stopping thresholds, which were evaluated only on development seeds before the frozen test.',
  'The straight-line baseline is intentionally weak on centre-blocking courses. Its measured 0% completion demonstrates that this generator demands avoidance. A motion-only planner with a preloaded obstacle map would answer a different question because it would have privileged access to geometry.')

chapter(10, 'Experiment Tracking and Hyperparameter Optimisation')
h('10.1 Study design and objective')
equation(6, 'J = 3G − 4C + mean(min(mᵢ, 0.8))')
p('G and C count goals and collisions among four tuning courses. mᵢ is the lower-bound minimum swept clearance in metres for course i. The clearance term breaks ties after outcome counts. Optuna sampled the empirical objective with a seeded tree-structured Parzen estimator and included four queued configurations.')
table('Optuna search and selected settings', ['Setting', 'Search interval', 'Selected value'], [
    ('Stop distance', '0.78–1.42 m', f'{config["settings"]["stop"]:.3f} m'),
    ('Yaw gain', '2.8–4.6', f'{config["settings"]["gain"]:.3f}'),
    ('Temporal window', '1–4 frames', config['settings']['window']),
    ('Lookahead', '3.0–5.0 m', f'{config["settings"]["lookahead"]:.3f} m'),
    ('Goal weight', '0.31–0.54', f'{config["settings"]["goal_weight"]:.3f}'),
    ('Protected half-width', '0.47–0.74 m', f'{config["settings"]["half_width"]:.3f} m')
], [.38,.31,.31])
figure('Optuna objective by trial on tuning seeds 200–203.', bar_chart(
    [f'Trial {t["trial"]}' for t in trials], [float(t['objective']) for t in trials], 13, lambda v:f'{v:.2f}'))
h('10.2 Evidence and validation')
p(f'Trial {study["best_trial"]} achieved J={study["objective"]:.4f}. This study took approximately {study["elapsed_s"]:.0f} seconds on the local CPU. The winning policy reached six of six validation goals against four of six for the default. The experiment included only six trials, so it provides limited evidence for model-based hyperparameter importance.',
  (f'MLflow experiment {tracking["experiment_id"]} records the actual frozen ablation and tuning evidence. The run IDs are {tracking["runs"]["ablation"]} and {tracking["runs"]["optuna"]}, respectively. The tracking script records model/source hashes, seed ranges, run parameters, trial objectives and metric artifacts.' if tracking else 'The MLflow import script is included. Review the exported CSV and JSON records for the measured study and ablation.'))
table('Observed tuning outcomes', ['Trial', 'Goals', 'Collisions', 'Mean clipped clearance (m)', 'Objective'], [
    (t['trial'],t['goals'],t['collisions'],f'{float(t["mean_clearance_m"]):.3f}',f'{float(t["objective"]):.4f}')
    for t in trials
], [.12,.13,.16,.38,.21])
p('The trial with the longest tested lookahead and high steering gain recorded one collision. Trial 5 and trial 1 each finished all four tuning layouts, with less than 0.002 m difference in mean clipped clearance. Selecting the empirical winner therefore required a distinct validation check, and the tuned setting was frozen before the 50-course test.')

chapter(11, 'Model Evaluation')
h('11.1 Split and metrics')
figure('Seed partitions used for fitting and held-out evaluation.', diagram(['100–111\nCalibration','200–203\nOptuna tuning','300–305\nValidation','1000–1049\nFrozen test']))
equation(7, 'AbsRel = mean(|d̂ᵢ − dᵢ| / dᵢ),    RMSE = √mean((d̂ᵢ − dᵢ)²)')
equation(8, 'δₖ = mean(1[max(d̂ᵢ/dᵢ, dᵢ/d̂ᵢ) < 1.25ᵏ]),   k = 1, 2')
p('The terms use valid pixel estimates and labels only. MSE and MAE are the average squared and absolute differences in metres squared and metres respectively. NumPy computes the metrics per image and averages images equally. The computed RMSE is therefore the mean of per-image RMSE. Saturated distances, sparse missing values and crop exclusions have recorded valid-pixel counts.')
benchmark = [*depth,kitti] + ([nyu] if nyu else [])
table('Depth metrics by source and alignment', ['Source', 'Images', 'δ₁', 'δ₂', 'AbsRel', 'RMSE (m)'], [
    (label, row['images'], *[f'{row[key]:.3f}' for key in ('delta1','delta2','absrel','rmse')])
    for label,row in zip(['Ray-cast median','Ray-cast fixed','KITTI median','NYU median'],benchmark)
], [.35,.12,.12,.12,.13,.16])
p('The estimates vary across domains. Sparse KITTI depth and crop select different pixels from the dense NYU and synthetic labels. Appearance, camera geometry and underlying scene distributions also differ. This comparison indicates an evaluation-domain gap without identifying a single cause.')
if nyu:
    p(f'NYU median alignment produced δ₁={nyu["delta1"]:.3f} on all 654 test frames, below the 0.80 requirement. No test frames were used to train the model or tune the controller in this project.')
if nyu_affine and nyu_affine['alignment'] == 'per_image_gt_inverse_affine':
    p(f'An extra NYU diagnostic used each frame’s valid label pixels to fit both inverse-depth scale and offset. It reached δ₁={nyu_affine["delta1"]:.3f} and AbsRel={nyu_affine["absrel"]:.3f}. That result measures what the model could recover with an additional label-derived fit and uses a different alignment protocol from the acceptance test.')
h('11.2 Matched 50-course ablation')
figure('Collision-free completion percentage in the local simulator.', bar_chart(
    ['MiDaS-small','Ground truth','Straight-line'], [mid['goals']*2, oracle['goals']*2, base['goals']*2], 100, lambda v:f'{v:.0f}%'))
table('Mission and safety outcomes', ['Arm', 'Goal count', '95% Wilson', 'Collisions', 'Stop-only collisions'], [
    (label,f'{row["goals"]}/{row["runs"]}',f'{row["wilson95"][0]:.1%}–{row["wilson95"][1]:.1%}',
     row['collisions'],f'{row["safety"]["collided"]}/20' if row.get('safety') else 'Outside arm')
    for label,row in [('MiDaS',mid),('Ground truth',oracle),('Straight-line',base)]
], [.23,.15,.23,.16,.23])
equation(9, 'Wilson centre = (p̂ + z²/(2n)) / (1 + z²/n)')
equation(10, 'Wilson half-width = z √(p̂(1−p̂)/n + z²/(4n²)) / (1 + z²/n)')
p(f'In (9) and (10), p̂=k/n is the observed completion fraction, k is the number of goals, n is the number of courses and z=1.96 estimates the central 95% interval. For MiDaS, k=46 and n=50. The point estimate meets the numerical target in this simulator, with an interval from {mid["wilson95"][0]:.1%} to {mid["wilson95"][1]:.1%}. The 90% lower-bound claim is unsupported.')
h('11.3 Flight and failure evidence')
close_cases = [r for r in runs if r['mode']=='midas' and (
    r['status']=='collision' or (r['safety']=='False' and float(r['clearance_m'])<.15))]
near = sum(r['status']=='goal' for r in close_cases)
p(f'MiDaS successful runs had a median goal time of {mid["time_to_goal_s"]["p50"]:.2f} s. Median minimum clearance was {mid["clearance_m"]["p50"]:.3f} m, across all 50 runs. The reviewed set comprises 4 navigation collisions, 14 stop-only collisions and {near} successful navigation runs with less than 0.15 m minimum clearance.',
  'Stop-only runs expose the sensing failure directly because the heading stays fixed. When the estimated frontal return is too distant or projected outside the protected width, the threshold can leave forward motion enabled as independent clearance vanishes. Normal runs may instead fail at the side of an obstacle during a turn. Texture, obstacle width and distance are candidate mechanisms. This evidence supports an observed mismatch between commands and clearance without proving which model feature caused it.')
if (WORK / 'failure_review.csv').exists():
    categories = Counter(r['event_category'] for r in read_csv('failure_review.csv'))
    p(f'At the lowest-clearance step in each flagged case, the oracle would have stopped while MiDaS continued in {categories["oracle_stop_only"]} cases. The case CSV keeps seed, timestamp, geometric clearance, both frontal depths and the actual command.')
figure(f'Seed {replays[0]["outcome"]["seed"]}, actual RGB (left) and MiDaS input to control (right).',
       replay_figure(replays[0], min(8, len(replays[0]['frames'])-1)))
if (WORK / 'failure_review.csv').exists():
    reviewed = read_csv('failure_review.csv')
    example_rows = [[r['seed'], r['status'], r['evidence_time_s'],
                     f'{float(r["geometric_clearance_m"]):.3f}',
                     f'{float(r["estimated_frontal_m"]):.2f}' if r['estimated_frontal_m'] else 'No frontal ray',
                     f'{float(r["oracle_frontal_m"]):.2f}' if r['oracle_frontal_m'] else 'No frontal ray']
                    for r in reviewed if r['arm']=='stop_only'][:5]
    table('Example stop-only collision steps, selected by seed order',
          ['Seed','Outcome','Time (s)','Clearance (m)','MiDaS front (m)','Oracle front (m)'],
          example_rows,[.13,.17,.13,.17,.20,.20])

chapter(12, 'Model Deployment and Application Integration')
h('12.1 Packaging and command interfaces')
figure('ROS 2 / Gazebo integration design for the Linux run.', diagram(['Gazebo RGB-D\nand odometry','ros_gz bridge\nto ROS 2','Depth node +\nreactive node','Twist command\nback to Gazebo']))
table('ROS topic interfaces', ['Topic', 'Payload', 'Role'], [
    ('/drone/camera/image', 'sensor_msgs/Image', 'RGB input for MiDaS and the diagnostic buffer.'),
    ('/drone/estimated_depth', 'sensor_msgs/Image', 'Float32 estimated axial depth, preserving the image stamp.'),
    ('/drone/camera/depth_image', 'sensor_msgs/Image', 'Ground truth for oracle and paired capture.'),
    ('/drone/odom', 'nav_msgs/Odometry', 'Position, yaw and separate track clearance.'),
    ('/drone/cmd_vel', 'geometry_msgs/Twist', 'World-frame linear velocity and yaw rate through the bridge.')
], [.38,.26,.36])
p('Portable execution uses Python, NumPy, OpenCV and ONNX Runtime. The reference assets are pinned by version and the released model by SHA-256. A Docker Compose service targets ROS 2 Humble and Gazebo Fortress on Linux [5]. The SDF world generates static obstacles and a velocity-controlled, fixed-altitude quadrotor representation. Integration execution requires a Linux host with Docker.')
h('12.2 Latency and serving decision')
backend_labels = {'onnx': 'ONNX Runtime CPU', 'torch': 'PyTorch FP32 CPU', 'tensorrt': 'TensorRT FP16'}
profrows = [(backend_labels[v['backend']], f"{v['mean_inference_fps']:.1f}", f"{v['loop_ms']['p50']:.1f}",
             f"{v['loop_ms']['p95']:.1f}", f"{v['loop_ms']['p99']:.1f}", str(v['over_100_ms']))
            for v in profiles if v]
if not profiles[2]:
    profrows.append(('TensorRT FP16','Pending','—','—','—','—'))
table('Serving profiles on 100 shared frames per backend', ['Backend', 'Depth FPS', 'p50 ms', 'p95 ms', 'p99 ms', '>100 ms'],
      profrows, [.26,.15,.14,.14,.14,.17])
a = profiles[0]
p(f'Each profile excludes ten warm-up frames. Capture, depth, policy and command are timed with a monotonic clock. ONNX median stage times were {a["capture_ms"]["p50"]:.2f}, {a["depth_ms"]["p50"]:.2f}, {a["policy_ms"]["p50"]:.2f} and {a["command_ms"]["p50"]:.4f} ms respectively. Computation throughput is distinct from the scheduler’s 10 Hz simulated time step and from transport latency. The frozen-run timing measures the local camera-to-command path without ROS/DDS and Gazebo overhead.')
p(f'The frozen MiDaS course run logged {mid["over_100_ms"]} of {mid["frame_count"]:,} loop measurements above 100 ms. The measured throughput meets the 10 FPS target. This occurrence rules out a hard worst-case deadline claim.')
if profiles[2]:
    p(f'TensorRT FP16 used the local {profiles[2].get("gpu", "NVIDIA GPU")}. The {profiles[2].get("build_seconds",0):.1f} s native engine build is outside steady-state timing. The course brief specifies a Colab GPU. Its timing remains pending. ONNX Runtime CPU remains the portable default because it gives sufficient typical throughput and simplifies installation.')
    p(f'On the common profile frames, median-aligned δ₁ was {a["median_aligned_delta1"]:.3f} for ONNX CPU and {profiles[2]["median_aligned_delta1"]:.3f} for TensorRT FP16. Their AbsRel values were {a["median_aligned_absrel"]:.3f} and {profiles[2]["median_aligned_absrel"]:.3f}. These measures describe observed agreement on this sample and do not establish per-pixel equivalence.')
else:
    p('ONNX Runtime CPU is the selected portable setup. Its loop distribution fits the local 100 ms budget. TensorRT requires a compatible NVIDIA GPU and CUDA/TensorRT runtime. The profiling harness exposes the backend and rejects CPU fallback. A verified TensorRT timing is pending.')
h('12.3 Stage breakdown and integration test criteria')
table('Per-stage latency p95 on the common 100-frame profile',
      ['Backend', 'Capture (ms)', 'Depth (ms)', 'Policy (ms)', 'Command (ms)'], [
    (backend_labels[v['backend']], *[f'{v[key]["p95"]:.3f}' for key in
                                    ('capture_ms','depth_ms','policy_ms','command_ms')])
    for v in profiles if v
], [.28,.18,.18,.18,.18])
p('A Gazebo test should compare the time stamped on a published camera image to the command publication time on the same simulation clock. It should also count the number of published frames and accepted commands to detect timeouts and dropped inputs. When the bridge is measured, record the real-time factor alongside simulated-clock latency because slower-than-real-time simulation affects interpretation.')

if gazebo:
    h('12.4 Recorded Gazebo demonstration')
    p('Gazebo Fortress and ROS 2 Humble ran in a two-core Ubuntu 22.04 GitHub Codespace. A single forward RGB image fed MiDaS v2.1 Small through ONNX Runtime CPU. The ROS 2 controller converted estimated depth to velocity commands. The run used a target real-time factor of 0.15 to give CPU inference time to keep up. The companion MP4 records the Gazebo desktop and plays it back at 1.5 times the recorded speed.')
    table('Recorded MiDaS Gazebo run', ['Measurement', 'Result'], [
        ('Course and seed', f'{gazebo["scene"]}, seed {gazebo["seed"]}'),
        ('Outcome', gazebo['status'].capitalize()),
        ('Elapsed simulation time', f'{gazebo["elapsed_sim_s"]:.2f} s'),
        ('Control frames and movement commands', f'{gazebo["frames"]} frames, {gazebo["motion_commands"]} commands'),
        ('Minimum geometric clearance', f'{gazebo["min_clearance_m"]:.3f} m'),
        ('Camera stamp to command, p95', f'{gazebo["latency_p95_sim_ms"]:.1f} ms on the simulated clock')
    ], [.51,.49])
    p('This single run verifies the ROS 2 and Gazebo integration at the configured pace. The reported latency uses the simulation clock. Wall-clock performance and the 50-course Gazebo acceptance target still require separate measurement. The archive includes the original run summary and per-frame telemetry.')
    if args.gazebo_image:
        figure(f'Gazebo Fortress view of the recorded Grove course, seed {gazebo["seed"]}.',
               Image(str(args.gazebo_image), width=W, height=W * 800 / 1280))

chapter(13, 'Monitoring, Drift and Continuous Integration')
h('13.1 Drift design and metric')
p('Data drift changes the distribution of observed descriptors. Fog injection manipulates appearance while preserving geometry. Concept drift would change the relation between appearances and the target, for example when a novel material creates systematically unreliable depth. These cases have different evidence requirements. A feature alert alone is insufficient to establish a change in task performance.')
equation(11, 'PSI = Σⱼ (cⱼ − rⱼ) ln(cⱼ/rⱼ)')
p('For bin j, rⱼ is the reference proportion and cⱼ is the current proportion. The monitor uses up to eight reference-quantile bins with additive 0.5 smoothing, which keeps empty counts finite. A value above 0.25 on two of the three configured features triggers degraded mode. The auxiliary contrast descriptor is diagnostic. Evidently’s report provides a separate dashboard with its own binning.')
table('Drift metrics from matched clean and foggy frames', ['Feature', 'PSI', 'Decision'], [
    (key.replace('_',' ').title(),f'{value:.3f}','Triggered' if key in drift['alerts'] else 'Diagnostic' if key=='contrast' else 'Below threshold')
    for key,value in drift['psi'].items()
], [.40,.20,.40])
p(f'There were {drift["per_condition"]} frames per condition. The trigger reduced the maximum speed to {drift["degraded_max_speed_mps"]:.1f} m/s and increased the threshold to {drift["degraded_stop_m"]:.2f} m. In the single foggy showcase the degraded controller timed out at 19 s with positive geometric clearance. Normal mode reached the goal on that pose seed. This test establishes that the fallback is connected and changes behaviour. It provides no population-level benefit estimate.')
h('13.2 Integration quality gate')
figure('Configured CI sequence for portable and Gazebo runs.', diagram([
    'Push or\npull request','Unit tests and\nportable smoke','Container launch\nand camera check','Five Gazebo\ncourse results']))
p('The local suite checks deterministic generation, valid camera outputs, stop behaviour, metric correctness and SDF structure. Five portable oracle smoke courses were executed. The GitHub Actions workflow adds a Docker build and a separate headless Gazebo job, checking movement, frames observed, positive clearance and goal outcomes. No hosted Actions run is included.',
  'A multi-window alert should prompt data quality inspection and labelled evaluation before any fine-tuning. When new weights become available, log their checksum, recompute parity and timing, validate on development scenes and reserve new frozen seeds for acceptance.')
table('Measured one-course drift intervention', ['Condition', 'Outcome', 'Time (s)', 'Minimum clearance (m)'], [
    (label, r['status'],f'{r["time_s"]:.1f}',f'{r["clearance_m"]:.3f}')
    for label,r in zip(('Clean, normal control','Fog, normal control','Fog, degraded control'),drift['scenarios'])
], [.38,.18,.18,.26])
p('The alert and state change were demonstrated during offline monitoring. The live ROS node supports a degraded-mode flag. A continuous watcher that feeds the state into a running ROS graph remains a deployment integration step.')

chapter(14, 'Responsible AI and Explainability')
h('14.1 Safety case and limits')
table('Hazard analysis and evidence', ['Hazard', 'Control in project', 'Residual risk or required evidence'], [
    ('Contact with property or a person', 'Protected corridor, velocity override, collision logging.', 'MiDaS failed 14 of 20 stop-only cases. Independent ranging and alternative sensing require validation.'),
    ('Stale frames or invalid estimates', 'Depth validity check and ROS watchdog for stale odometry or depth.', 'Verify timeout under real transport pressure and braking delay.'),
    ('Clearance error while turning', 'Independent swept clearance from course geometry.', 'Fixed altitude and low-order dynamics omit rotor and wind effects.'),
    ('Bystander capture or surveillance', 'Synthetic data and civil inspection scope.', 'Use on-device processing, no-retention defaults and local access controls before any future collection.'),
    ('Unauthorised airspace entry', 'Aircraft operation outside the project.', 'Review applicable aviation requirements and geofencing for future field work.')
], [.29,.33,.38])
h('14.2 Interpretable control and case review')
p('The policy exposes frontal depth, yaw command, speed and clearance. These records let a reviewer distinguish the stop threshold, estimated distance, geometry and chosen direction at an individual frame. A controlled counterfactual is the matched oracle intervention: changing the depth source while maintaining the course and policy changes four failed navigation outcomes to completed courses and eliminates stop-only contacts on these seeds.',
  'The intervention explains behavioural sensitivity to perception and does not attribute error to a specific RGB feature. SHAP and DiCE were not used. The package includes camera and depth replays for inspecting what the policy received next to the path the drone took.')
h('14.3 Governance and intended use')
p('The prototype is simulation-only and carries no flight certification. For a future system, the EU AI Act’s safety-component classification is conditional on the Article 6 criteria and the relevant product legislation [10]. Civil aviation and unmanned aircraft have sector-specific provisions. The educational documentation uses risk records, test traces and human review as practice. It makes no legal classification or compliance determination for a product.',
  'The intended use is teaching perception and safety evaluation for simulated civil inspection. Real-world autonomous pursuit, surveillance and covert recording are outside the intended use. A future operational review would require a responsible operator, incident reporting and a validated safe fallback.')
table('Evidence gates for future safety review', ['Claim', 'Required evidence'], [
    ('Stop command arrives in time', 'Measure latency including transport and actuator delay, then compare stopping distance with calibrated clearance and attitude.'),
    ('Fallback works after perception loss', 'Inject stale frames, dropped odometry and invalid depth. Confirm the independent fallback preserves a safe state.'),
    ('Changes preserve safety', 'Test new weights and simulator scenes against fresh acceptance seeds and tracked human review.'),
    ('Privacy remains bounded', 'Document purpose, permitted retention, access logging and no-recording defaults before any trial with real imagery.')
], [.38,.62])

chapter(15, 'Project Management, Results and Closure')
h('15.1 Milestones and contribution record')
table('Planned semester milestones and verified status', ['Weeks', 'Milestone', 'Implementation evidence'], [
    ('1–3','Scope and quantitative requirements','Scope, stakeholder matrix and timing derivation.'),
    ('4–5','Simulation and data quality','Ray-cast worlds, 5,000 pairs, manifest and EDA. Gazebo worlds authored.'),
    ('6–9','MiDaS, tuning and evaluation','Model inference, six Optuna trials, frozen ablation and stop-only experiments.'),
    ('10','Packaging and latency','CPU profiles, source package and Docker setup. ' +
     ('Local TensorRT profile.' if profiles[2] else 'TensorRT profile pending.')),
    ('11–12','Monitoring and CI','Fog injection and portable smoke checks. Hosted CI execution remains open.'),
    ('13–15','Safety review and presentation','Risk analysis, model and data cards, cloud Gazebo demonstration and defense deck.' if gazebo else 'Risk analysis, model and data cards, portable replay and defense deck.')
], [.13,.38,.49])
table('Contribution record for group completion', ['Workstream', 'Person and evidence to confirm'], [
    ('Infrastructure and simulation','[Name] • World generator, simulator, Gazebo runtime verification'),
    ('Perception and benchmarks','[Name] • Checkpoint, calibration and depth metrics'),
    ('Controls and evaluation','[Name] • Reactive policy, tuning, ablation and failure review'),
    ('MLOps and documentation','[Name] • Tracking, monitoring, report and presentation')
], [.35,.65])
p('The group should replace each proposed assignment with actual named contributions. The draft includes no completed signatures, attendance records or faculty approvals.')
h('15.2 Acceptance and project closure')
table('Requirements status at handoff', ['Requirement', 'Evidence', 'Status'], [
    ('NFR1 ≥90% estimated depth in Gazebo', '46/50 in the local ray-cast simulator. One recorded Gazebo integration flight.' if gazebo else '46/50 in the local ray-cast simulator.', 'Gazebo batch pending'),
    ('NFR2 ≥10 FPS and NFR3 ≤100 ms', f'ONNX profile {a["mean_inference_fps"]:.1f} depth FPS. Frozen loop p95 {mid["latency"]["loop_ms"]["p95"]:.1f} ms with {mid["over_100_ms"]} deadline miss.', 'Throughput met. Deadline miss observed.'),
    ('NFR4 Zero stop-only contacts', '14/20 with estimated depth, 0/20 in oracle arm.', 'Failed'),
    ('NFR5 NYU δ₁ ≥0.80', f'{nyu["delta1"]:.3f} on {nyu["images"]} evaluated frames.' if nyu else 'No measured NYU result included.',
     ('Met' if nyu['delta1'] >= .8 and nyu['images'] == 654 else 'Failed') if nyu else 'Pending'),
    ('NFR6 Reproducibility', 'Versioned seeds and dependencies, runner, manifest and experiment outputs.', 'Portable reproduction path')
], [.32,.46,.22])
p('The implementation has a runnable portable prototype and a measured navigation baseline. ' + (
    'The cloud Gazebo demonstration verifies one integrated flight at 15% of real time. The Gazebo batch, physical stopping behaviour and worst-case wall-clock latency remain open.' if gazebo else
    'Gazebo integration and its transport latency remain verification activities.'))

chapter(16, 'References')
for i, (title, link) in enumerate([
    ('R. Ranftl et al., “Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer,” IEEE TPAMI, 2022. MiDaS model releases and code.', 'https://github.com/isl-org/MiDaS'),
    ('N. Silberman et al., “Indoor Segmentation and Support Inference from RGBD Images,” ECCV, 2012. NYU Depth V2.', 'https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html'),
    ('A. Geiger, P. Lenz, and R. Urtasun, “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite,” CVPR, 2012. Raw data and calibration.', 'https://www.cvlibs.net/datasets/kitti/raw_data.php'),
    ('C. Godard et al., “Digging Into Self-Supervised Monocular Depth Estimation,” ICCV, 2019. Public Eigen evaluation file list used for sample selection.', 'https://github.com/nianticlabs/monodepth2/tree/master/splits/eigen'),
    ('Open Robotics, “Installing Gazebo with ROS,” Gazebo Fortress documentation.', 'https://gazebosim.org/docs/fortress/ros_installation/'),
    ('Microsoft, “ONNX Runtime Execution Providers.”', 'https://onnxruntime.ai/docs/execution-providers/'),
    ('IBM, “CRISP-DM Help Overview.”', 'https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=dm-crisp-help-overview'),
    ('Microsoft, “Team Data Science Process lifecycle detail.”', 'https://github.com/Azure/Microsoft-TDSP/blob/master/Docs/lifecycle-detail.md'),
    ('Google Cloud, “MLOps: Continuous delivery and automation pipelines in machine learning.”', 'https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning'),
    ('European Union, Regulation (EU) 2024/1689, Artificial Intelligence Act.', 'https://eur-lex.europa.eu/eli/reg/2024/1689'),
    ('European Union, Regulation (EU) 2016/679, General Data Protection Regulation.', 'https://eur-lex.europa.eu/eli/reg/2016/679')
],1):
    story.append(paragraph(f'[{i}] {clean(title)} <link href="{link}" color="#1b685a">Online source</link>', REF, markup=True))
if nyu and nyu.get('mirror'):
    p('The NYU run used the mirror named in the project’s machine-readable provenance record.')

page()
front('Appendix A. Reproduction and submission checklist')
p('Extract the source archive and open README.md for commands. Run the supplied browser replay separately by opening its HTML file. The Python live viewer can generate an additional seed on demand. For local reproduction, initialise the pinned Python dependencies, download the official weights and use the shipped calibration and settings.')
p('Inspect results/courses.csv for one row per course and results/frames.csv.gz for the full frame telemetry. The summary JSON contains derived metrics and Wilson bounds. Reported case failures can be inspected by seed. The replay displays geometric clearance separately from model frontal depth.')
story.append(paragraph('The GitHub Codespaces setup is available at <link href="https://github.com/AryanBhimani15/drone-gazebo-demo" color="#1b685a">drone-gazebo-demo</link>. The repository uses private GitHub access. The full project source, recorded Gazebo demonstration, and evaluation files are included in the submission archive.', markup=True))
p('Complete the institutional cover information and individual contributions, and obtain the required signatures before submission.')
front('Appendix B. Data and model card summary')
p('The model artifact is the official MiDaS v2.1 Small ONNX export. It receives RGB and returns relative inverse depth. The package uses fixed project calibration for driving commands and per-image median scaling with labelled depth for offline evaluation. Approximate metric output can be severely wrong within the stopping envelope.')
p('The synthetic data are generated from seeded static scenes. Camera intrinsics, export version and content hashes accompany the 5,000-frame manifest. Benchmark sources and licensing pointers are in the repository. The experimental domain excludes moving obstacles, outdoor flight dynamics and human subjects.')

doc = Report(str(OUTPUT), pagesize=A4, leftMargin=72, rightMargin=72,
             topMargin=72, bottomMargin=72, title='Autonomous Drone Obstacle Avoidance Using Monocular Depth Estimation',
             author='B.Tech AI & ML Project 6')
doc.addPageTemplates(PageTemplate(id='A4', frames=[Frame(72,72,W,HEIGHT-144,
    leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)], onPage=footer))
doc.multiBuild(story)
