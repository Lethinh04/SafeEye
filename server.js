const express = require("express");
const path    = require("path");

const app  = express();
const PORT = process.env.PORT || 3000;

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
