const path = require("path");
const express = require("express");
const cors = require("cors");
require("dotenv").config();

const authRouter = require("./routes/auth");

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use("/api", authRouter);
app.use(express.static(path.join(__dirname, "..", "frontend")));

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "express-login-system" });
});

app.listen(port, () => {
  console.log(`Login system running at http://localhost:${port}`);
});