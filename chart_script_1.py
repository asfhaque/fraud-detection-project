import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Ensure static image export works
try:
    import kaleido  # noqa: F401
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaleido"])

# Provided data
data = {
    "Amount": [288.72, 3.51, 8.01, 12.21, 44.22, 15.45, 25.33, 67.89, 156.78, 89.45, 234.56, 45.67, 78.90, 123.45, 567.89, 34.56, 78.90, 234.56, 456.78, 890.12, 123.45, 67.89, 234.56, 45.67, 356.78, 789.01, 234.56, 567.89, 345.67, 123.45],
    "Time": [19658.03, 126667.89, 116295.97, 134673.97, 167946.76, 87546.32, 145673.89, 65432.10, 123456.78, 98765.43, 156789.01, 87654.32, 109876.54, 134567.89, 167890.12, 76543.21, 145632.87, 189076.54, 198765.43, 156432.10, 87546.32, 145673.89, 65432.10, 123456.78, 98765.43, 156789.01, 87654.32, 109876.54, 134567.89, 167890.12],
    "Class": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# Use brand colors with better contrast - cyan for normal, red for fraud
colors = {0: "#1FB8CD", 1: "#B4413C"}

fig = go.Figure()
for cls, label in zip([0, 1], ["Normal", "Fraud"]):
    subset = df[df["Class"] == cls]
    fig.add_trace(go.Scatter(
        x=subset["Time"],
        y=subset["Amount"],
        mode="markers",
        marker=dict(color=colors[cls], size=8),
        name=label,
        cliponaxis=False,
        hovertemplate="<b>%{fullData.name}</b><br>Time: %{x}<br>Amount: $%{y}<extra></extra>"
    ))

fig.update_layout(
    title="Transaction Time vs Amount by Class",
    xaxis_title="Time",
    yaxis_title="Amount ($)",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_xaxes(tickformat="~s")
fig.update_yaxes(tickformat="$~s")

fig.write_image("fraud_detection_scatter.png")