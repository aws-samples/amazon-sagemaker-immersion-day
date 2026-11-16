"""Generate the model build architecture diagram of the SageMaker Projects lab.

Run from this folder:

    python design/build_diagrams.py

The style follows img/deployment-project-architecture.svg, the deployment diagram of the
amazon-sagemaker-from-idea-to-production workshop (notebook 05), which this lab reuses as is.
The icons in design/icons come from the AWS Architecture Icons asset package
(https://aws.amazon.com/architecture/icons/) and are embedded as base64 data URIs,
so the generated SVG is a single self-contained file.
"""
import base64
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICONS = HERE / "icons"
OUT = HERE.parent / "img" / "model-build-project-architecture.svg"

# Accent colors of the deployment diagram
SAGEMAKER = "#00A4A6"
STORAGE = "#7AA116"
APP_INTEGRATION = "#DD344C"
DEVELOPER_TOOLS = "#0073BB"
GENERAL = "#232F3E"

STYLE = """
    text { font-family: 'Amazon Ember', Arial, Helvetica, sans-serif; fill: #37475A; }
    .heading { font-size: 31px; font-weight: 700; fill: #232F3E; }
    .group-label { font-size: 24px; font-weight: 700; fill: #232F3E; }
    .node { fill: #FFFFFF; stroke: #AAB7B8; stroke-width: 2; }
    .node-label { font-size: 22px; font-weight: 700; fill: #232F3E; }
    .node-detail { font-size: 17px; font-weight: 400; fill: #687078; }
    .code { font-family: 'Courier New', monospace; font-size: 17px; font-weight: 700; fill: #232F3E; }
    .flow-label { font-size: 17px; font-weight: 600; fill: #687078; paint-order: stroke; stroke: #FFFFFF; stroke-width: 7px; stroke-linejoin: round; }
    .small-label { font-size: 16px; font-weight: 600; fill: #687078; }
    .arrow { fill: none; stroke: #687078; stroke-width: 3; marker-end: url(#arrow); }
    .group { fill: #F9FAFB; stroke: #7D8998; stroke-width: 3; }
    .subgroup { fill: #FFFFFF; stroke: #0073BB; stroke-width: 3; }
"""


def icon_uri(name):
    return "data:image/svg+xml;base64," + base64.b64encode((ICONS / f"{name}.svg").read_bytes()).decode()


parts = ['<rect width="100%" height="100%" fill="#FFFFFF"/>']
add = parts.append


def text(x, y, value, cls, anchor="start"):
    add(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>')


def lines(x, y, values, cls, step):
    spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else step}">{escape(v)}</tspan>' for i, v in enumerate(values))
    add(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="middle">{spans}</text>')


def image(name, x, y, size):
    add(f'<image href="{icon_uri(name)}" x="{x}" y="{y}" width="{size}" height="{size}" preserveAspectRatio="xMidYMid meet"/>')


def group(x, y, w, h, label, icon, cls="group", stroke=None):
    style = f'fill="#FFFFFF" stroke="{stroke}" stroke-width="3"' if stroke else f'class="{cls}"'
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" {style}/>')
    image(icon, x + 25, y + 21, 48)
    text(x + 88, y + 53, label, "group-label")


def node(x, y, w, h, label, detail, icon, accent, code=None):
    """Same node as the deployment diagram: accent bar, 64 px icon, bold label, gray detail."""
    cx = x + w / 2
    add("<g>")
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" class="node"/>')
    add(f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{accent}"/>')
    image(icon, cx - 32, y + 24, 64)
    label = [label] if isinstance(label, str) else label
    lines(cx, y + 108, label, "node-label", 25)
    cursor = y + 108 + 25 * (len(label) - 1)
    if code:
        cursor += 28
        text(cx, cursor, code, "code", "middle")
    lines(cx, cursor + 40, detail, "node-detail", 20)
    add("</g>")


def arrow(d, label=None, lx=None, ly=None, anchor="middle", dashed=False):
    add(f'<path d="{d}" class="arrow"{" stroke-dasharray=\"8 7\"" if dashed else ""}/>')
    if label:
        text(lx, ly, label, "flow-label", anchor)


# Groups: the project toolchain on the left, the model registry and MLflow on the right
group(30, 55, 1410, 955, "SageMaker Project: model build toolchain", "Arch_Amazon-SageMaker_64")
group(1470, 55, 300, 955, "Registry, MLflow", "Arch_Amazon-SageMaker_64", stroke=SAGEMAKER)

# Top row: data scientist, repository, rule, artifact bucket
node(60, 150, 210, 215, "Data scientist", ["Changes the", "step code"], "Res_Users_48_Light", GENERAL)
node(360, 150, 230, 215, "AWS CodeCommit", ["Model build code", "main branch"], "Arch_AWS-CodeCommit_64", DEVELOPER_TOOLS)
node(760, 150, 230, 215, ["Amazon", "EventBridge"], ["Project rule"], "Arch_Amazon-EventBridge_64", APP_INTEGRATION)
node(1150, 150, 230, 215, "Amazon S3", ["Pipeline artifacts"], "Arch_Amazon-Simple-Storage-Service_64", STORAGE)
arrow("M 270 258 L 360 258", "git push", 315, 235)
arrow("M 590 258 L 760 258", "main branch update", 675, 235)
arrow("M 875 365 L 875 420", "start execution", 905, 402, "start")

# CodePipeline model build workflow
add('<rect x="75" y="420" width="1320" height="240" rx="16" class="subgroup"/>')
image("Arch_AWS-CodePipeline_64", 100, 442, 48)
text(163, 474, "AWS CodePipeline model build workflow", "group-label")
node(105, 495, 200, 145, "Source", [], "Arch_AWS-CodeCommit_64", DEVELOPER_TOOLS)
node(335, 495, 260, 145, "Build", [], "Arch_AWS-CodeBuild_64", DEVELOPER_TOOLS)
arrow("M 305 568 L 335 568")
text(640, 540, "codebuild-buildspec.yml", "code")
text(640, 568, "installs requirements.txt and runs pipeline.py:", "node-detail")
text(640, 592, "upserts the SageMaker pipeline with the project", "node-detail")
text(640, 616, "role, MLflow app, image, and model package group,", "node-detail")
text(640, 640, "and starts an execution", "node-detail")
arrow("M 1265 365 L 1265 400 L 1340 400 L 1340 420", "artifact store", 1300, 390, dashed=True)

# SageMaker pipeline
add(f'<rect x="75" y="705" width="1320" height="275" rx="16" fill="#FFFFFF" stroke="{SAGEMAKER}" stroke-width="3"/>')
image("Arch_Amazon-SageMaker_64", 100, 727, 48)
text(163, 759, "SageMaker pipeline sagemaker-<project name>", "group-label")
text(1370, 759, "ml.r5.xlarge, warm pool", "small-label", "end")
arrow("M 465 640 L 465 705", "upsert + start", 495, 680, "start")
steps = [
    ("Preprocess", ["Scale, one-hot,", "split the data"], "Arch_Amazon-SageMaker_64"),
    ("Train", ["XGBoost", "regression"], "Res_Amazon-SageMaker_Train_48"),
    ("Evaluate", ["Test MSE"], "Arch_Amazon-SageMaker_64"),
    ("Register", ["New model version"], "Res_Amazon-SageMaker_Model_48"),
]
for i, (label, detail, icon) in enumerate(steps):
    x = 105 + i * 320
    node(x, 780, 270, 180, label, detail, icon, SAGEMAKER)
    if i:
        arrow(f"M {x - 50} 870 L {x} 870")

# Registry and MLflow
node(1505, 150, 230, 230, ["SageMaker", "Model Registry"], ["Model package group", "<project>-<project ID>"], "Arch_Amazon-SageMaker_64", SAGEMAKER)
node(1505, 600, 230, 245, ["SageMaker", "MLflow app"], ["One run per pipeline", "version, nested", "run per step"], "Arch_Amazon-SageMaker_64", SAGEMAKER)
arrow("M 1335 870 L 1455 870 L 1455 330 L 1505 330")
text(1620, 412, "new version:", "small-label", "middle")
text(1620, 436, "PendingManualApproval", "small-label", "middle")
arrow("M 1395 735 L 1430 735 L 1430 705 L 1505 705", dashed=True)
text(1460, 690, "log", "flow-label", "middle")
text(1495, 935, "Approving a version starts", "small-label")
text(1495, 959, "the deployment pipeline.", "small-label")

svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1040" viewBox="0 0 1800 1040" role="img" aria-labelledby="title desc">\n'
    '<title id="title">SageMaker Project model build architecture</title>\n'
    '<desc id="desc">A data scientist pushes to the main branch of the model build CodeCommit repository. '
    'An EventBridge rule starts the model build CodePipeline pipeline. Its Build stage runs pipeline.py in CodeBuild, '
    'which upserts and starts the SageMaker pipeline: preprocess, train, evaluate, and register on ml.r5.xlarge with a warm pool. '
    'The register step adds a model version with the status PendingManualApproval to the project model package group, '
    'and every step logs to the MLflow app. Approving a version starts the deployment pipeline.</desc>\n'
    '<defs>\n'
    '  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#687078"/></marker>\n'
    f'  <style>{STYLE}  </style>\n'
    '</defs>\n'
    + "\n".join(parts)
    + "\n</svg>\n"
)
OUT.write_text(svg, encoding="utf-8")
print(f"Wrote {OUT}")
