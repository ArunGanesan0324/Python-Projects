const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();
app.use(cors());
app.use(express.json()); // To parse JSON requests

app.post("/predict", (req, res) => {
    const { domain, difficulty } = req.body;

    // Debugging: Check if domain and difficulty are received correctly
    console.log("Received request:", { domain, difficulty });

    if (!domain) {
        console.log("Error: Domain is required.");
        return res.status(400).json({ error: "Domain is required." });
    }

    // Debugging: Log that the prediction request is being processed
    console.log("Generating project ideas for domain:", domain, "and difficulty:", difficulty);

    // Spawn Python process with arguments
    const pythonProcess = spawn("python", ["project_idea_generator.py", domain, difficulty || ""]);

    let result = "";

    // Debugging: Check if Python process stdout is working
    pythonProcess.stdout.on("data", (data) => {
        console.log("Python process output:", data.toString());
        result += data.toString();
    });

    pythonProcess.stderr.on("data", (error) => {
        // Debugging: Log Python errors (if any)
        console.error("Python process error:", error.toString());
    });

    pythonProcess.on("close", (code) => {
        // Debugging: Log process completion and exit code
        console.log("Python process closed with code:", code);

        if (code !== 0) {
            console.error("Error: Python script failed with exit code", code);
            return res.status(500).json({ error: "Error generating project ideas." });
        }

        // Debugging: Log the final prediction result
        console.log("Prediction result:", result.trim());

        // Send the result back to the client
        res.json({ prediction: result.trim() });
    });
});

// Start server
app.listen(5000, () => {
    console.log("Server running on port 5000");
});
