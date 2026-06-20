const express = require("express");
const path    = require("path");
const admin   = require("firebase-admin");

const app  = express();
const PORT = process.env.PORT || 3000;

// Parse JSON bodies
app.use(express.json());

// Initialize Firebase Admin (server) if service account provided
let db = null;
if (process.env.FIREBASE_SERVICE_ACCOUNT) {
    try {
        const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount),
            databaseURL: "https://sos-app-8ba8b-default-rtdb.asia-southeast1.firebasedatabase.app"
        });
        db = admin.database();
        console.log("Firebase Admin initialized from env FIREBASE_SERVICE_ACCOUNT");
    } catch (err) {
        console.error("Invalid FIREBASE_SERVICE_ACCOUNT:", err.message);
    }
} else {
    try {
        const serviceAccount = require("./serviceAccountKey.json");
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount),
            databaseURL: "https://sos-app-8ba8b-default-rtdb.asia-southeast1.firebasedatabase.app"
        });
        db = admin.database();
        console.log("Firebase Admin initialized from serviceAccountKey.json");
    } catch (err) {
        console.warn("Firebase Admin not initialized. Add serviceAccountKey.json or set FIREBASE_SERVICE_ACCOUNT env var.");
    }
}

// EJS template engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "src/view"));

// Static files
app.use(express.static(path.join(__dirname, "public")));

// ─── Routes ───────────────────────────────
app.get("/", (req, res) => res.redirect("/demo"));

app.get("/demo", (req, res) => {
    res.render("demo", { title: "SafeEye – Demo AI" });
});

// Serve simple_test.html for quick testing
app.get('/simple_test', (req, res) => {
    res.sendFile(path.join(__dirname, 'simple_test.html'));
});

// API endpoint to receive detection payloads and push to Firebase Realtime Database
app.post("/api/detections", async (req, res) => {
    if (!db) return res.status(500).json({ error: "Firebase not configured on server" });
    const payload = req.body;
    try {
        const newRef = db.ref("detections").push();
        await newRef.set({ ...payload, timestamp: Date.now() });
        res.json({ ok: true, id: newRef.key });
    } catch (err) {
        console.error("Failed to push detection to Firebase:", err);
        res.status(500).json({ error: err.message });
    }
});

// ─── Start ────────────────────────────────
app.listen(PORT, () => {
    console.log(`\n┌─────────────────────────────────────────┐`);
    console.log(`│  SafeEye Web Server                      │`);
    console.log(`│  http://localhost:${PORT}/demo              │`);
    console.log(`│                                          │`);
    console.log(`│  ⚠ Đảm bảo chạy Python API trước:       │`);
    console.log(`│  py src/model/demo_api.py            │`);
    console.log(`└─────────────────────────────────────────┘\n`);
});
