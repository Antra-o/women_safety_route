from flask import Flask, render_template, request
import pandas as pd
import networkx as nx
import osmnx as ox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ---------- LOAD DATA ----------
df = pd.read_csv("women_dataset.csv")
df.columns = df.columns.str.strip()

# ---------- ENCODING ----------
df["Police_Presence"] = df["Police_Presence"].map({"Low":0,"Medium":1,"High":2})
df["Crowd_Density"] = df["Crowd_Density"].map({"Low":0,"Medium":1,"High":2})

# ---------- SAFETY SCORE ----------
def calculate_score(row):
    score = (
        row["Theft"]*0.15 +
        row["Assault"]*0.25 +
        row["Harassment"]*0.20 +
        row["Stalking"]*0.10 +
        row["Robbery"]*0.10 +
        row["Kidnapping"]*0.10 +
        row["Rape"]*0.30
    )

    if row["Police_Presence"] == 2:
        score *= 0.7
    elif row["Police_Presence"] == 1:
        score *= 0.85

    if row["Crowd_Density"] == 0:
        score *= 1.2

    return score

df["Safety_Score"] = df.apply(calculate_score, axis=1)

def label(score):
    if score < 10:
        return 0
    elif score < 20:
        return 1
    else:
        return 2

df["Zone"] = df["Safety_Score"].apply(label)

# ---------- MODEL ----------
X = df[[
    "Theft","Assault","Harassment","Stalking",
    "Robbery","Kidnapping","Rape",
    "Police_Presence","Crowd_Density"
]]

y = df["Zone"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------- ML RISK ----------
def predict_risk(row):
    features = pd.DataFrame([row[X.columns]])
    return model.predict(features)[0]

df["Predicted_Zone"] = df.apply(predict_risk, axis=1)

def risk_value(zone):
    return [5, 15, 30][zone]

df["ML_Risk"] = df["Predicted_Zone"].apply(risk_value)

# ---------- LOAD REAL ROAD NETWORK ----------
place = "Dehradun, Uttarakhand, India"
G = ox.graph_from_place(place, network_type='drive')

# ---------- GET NEAREST NODE ----------
def get_node(lat, lon):
    return ox.distance.nearest_nodes(G, lon, lat)

# ---------- GET RISK FOR LOCATION ----------
def get_risk(lat, lon):
    # find nearest dataset point
    nearest = df.iloc[((df["Latitude"]-lat)**2 + (df["Longitude"]-lon)**2).idxmin()]
    return nearest["ML_Risk"]

# ---------- ROUTE ----------
@app.route("/", methods=["GET","POST"])
def index():
    path = None
    route_coords = []

    if request.method == "POST":
        source = request.form["source"]
        destination = request.form["destination"]

        src = df[df["Location"] == source].iloc[0]
        dst = df[df["Location"] == destination].iloc[0]

        source_node = get_node(src["Latitude"], src["Longitude"])
        dest_node = get_node(dst["Latitude"], dst["Longitude"])

        try:
            # shortest path based on distance
            route = nx.shortest_path(G, source_node, dest_node, weight='length')

            # convert to coordinates
            for node in route:
                lat = G.nodes[node]['y']
                lon = G.nodes[node]['x']
                route_coords.append([lat, lon])

            path = [source, destination]

        except:
            path = ["No route found"]

    return render_template(
        "index.html",
        locations=df["Location"].tolist(),
        path=path,
        route_coords=route_coords
    )

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == "__main__":
    app.run(debug=True)