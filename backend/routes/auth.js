const express = require("express");

const router = express.Router();

const demoUser = {
  email: "demo@nexus.local",
  password: "demo1234",
  name: "Demo User"
};

router.post("/login", (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({
      ok: false,
      message: "Email and password are required."
    });
  }

  if (email !== demoUser.email || password !== demoUser.password) {
    return res.status(401).json({
      ok: false,
      message: "Invalid credentials."
    });
  }

  return res.json({
    ok: true,
    user: {
      email: demoUser.email,
      name: demoUser.name
    },
    token: "local-demo-token"
  });
});

module.exports = router;