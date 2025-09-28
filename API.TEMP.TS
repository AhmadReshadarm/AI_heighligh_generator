import { IncomingForm } from "formidable";
import { spawn } from "child_process";
import path from "path";
import fs from "fs/promises";
import fsSync from "fs";
import { NextApiRequest, NextApiResponse } from "next";

// Define the shape of the successful response data
interface SuccessResponse {
  highlightUrls: string[];
  message: string;
  pythonLogs: string;
}

// Define the shape of the error response data
interface ErrorResponse {
  message: string;
  error: string;
}

// Configuration for Next.js to disable body parsing for file upload
export const config = {
  api: {
    bodyParser: false,
    // Increased the internal Next.js request size limit to 6GB
    sizeLimit: "6gb",
  },
};

const PYTHON_SCRIPT_PATH = path.join(process.cwd(), "highlight_generator.py");
const VIDEO_UPLOAD_DIR = path.join(process.cwd(), "public", "temp");
const VIDEO_OUTPUT_DIR = path.join(process.cwd(), "public", "videos");

// Function to handle the formidable file upload parsing
const parseForm = (
  req: NextApiRequest
): Promise<{ videoPath: string; originalFilename: string }> => {
  return new Promise((resolve, reject) => {
    const form = new IncomingForm({
      uploadDir: VIDEO_UPLOAD_DIR,
      keepExtensions: true,
      // Increased formidable's maxFileSize to 6GB
      maxFileSize: 6 * 1024 * 1024 * 1024,
    });

    form.parse(req, (err, fields, files) => {
      if (err) {
        console.error("Formidable parse error:", err);
        return reject(err);
      }

      const videoFile = files.video?.[0];
      if (!videoFile) {
        return reject(new Error("No video file provided."));
      }

      resolve({
        videoPath: videoFile.filepath!,
        originalFilename: videoFile.originalFilename!,
      });
    });
  });
};

// Main API handler function
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<SuccessResponse | ErrorResponse>
) {
  if (req.method !== "POST") {
    return res.status(405).json({
      message: "Method Not Allowed",
      error: "Only POST is supported.",
    });
  }

  // Ensure directories exist
  await fs.mkdir(VIDEO_UPLOAD_DIR, { recursive: true });
  await fs.mkdir(VIDEO_OUTPUT_DIR, { recursive: true });

  let tempVideoPath: string | null = null;

  try {
    // 1. Parse the incoming video file
    const { videoPath, originalFilename } = await parseForm(req);
    tempVideoPath = videoPath;

    // Create a unique output directory for this job
    const uniqueId =
      path.parse(path.basename(originalFilename)).name + "_" + Date.now();
    const jobOutputDir = path.join(VIDEO_OUTPUT_DIR, uniqueId);
    await fs.mkdir(jobOutputDir, { recursive: true });

    console.log(
      `Starting Python script with input: ${tempVideoPath} and output: ${jobOutputDir}`
    );

    // 2. Spawn the Python process
    const pythonProcess = spawn("python", [
      PYTHON_SCRIPT_PATH,
      tempVideoPath, // Argument 1: Path to the uploaded video
      jobOutputDir, // Argument 2: Path where output videos should be saved
    ]);

    let pythonStdout = "";
    let pythonStderr = "";

    // Capture standard output and error streams
    pythonProcess.stdout.on("data", (data) => {
      const output = data.toString();
      pythonStdout += output;
      // You can see the Python logs streamed in real-time here
      // console.log("Python stdout chunk:", output);
    });

    pythonProcess.stderr.on("data", (data) => {
      const output = data.toString();
      pythonStderr += output;
      console.error("Python stderr chunk:", output);
    });

    // Wait for the Python process to finish
    const exitCode = await new Promise<number>((resolve) => {
      pythonProcess.on("close", resolve);
    });

    // 3. Check for successful exit
    if (exitCode !== 0) {
      console.error(`Python script exited with code ${exitCode}`);
      throw new Error(`Video processing failed. Python Logs: ${pythonStderr}`);
    }

    // 4. Extract structured data (JSON) from stdout using markers
    const startMarker = "---PYTHON-OUTPUT-START---\n";
    const endMarker = "\n---PYTHON-OUTPUT-END---";

    const startIndex = pythonStdout.indexOf(startMarker);
    const endIndex = pythonStdout.indexOf(
      endMarker,
      startIndex + startMarker.length
    );

    let highlightUrls: string[] = [];

    if (startIndex !== -1 && endIndex !== -1) {
      const jsonString = pythonStdout
        .substring(startIndex + startMarker.length, endIndex)
        .trim();

      console.log("Attempting to parse JSON from Python output:", jsonString);

      const absolutePaths: string[] = JSON.parse(jsonString);

      // 5. Convert absolute server paths to relative public URLs for the frontend
      highlightUrls = absolutePaths.map((absPath: string) => {
        // The path is relative to the 'public' directory
        const relativePath = path
          .relative(path.join(process.cwd(), "public"), absPath)
          .replace(/\\/g, "/"); // Normalize path separators for URL
        return "/" + relativePath;
      });
    } else {
      console.warn(
        "Could not find start/end markers in Python output. Output was:",
        pythonStdout
      );
      // We can still proceed, but highlightUrls will be empty, which might be an error.
    }

    // 6. Send the successful response to the Next.js frontend
    res.status(200).json({
      highlightUrls,
      message: "Processing complete. Highlights generated.",
      pythonLogs: pythonStderr,
    });
  } catch (error: unknown) {
    let errorMessage = "An unknown error occurred.";
    if (error instanceof Error) {
      errorMessage = error.message;
    } else if (typeof error === "string") {
      errorMessage = error;
    }

    console.error("API Error:", errorMessage);
    res
      .status(500)
      .json({ message: "Error processing video.", error: errorMessage });
  } finally {
    // 7. Cleanup the temporary input file
    if (tempVideoPath && fsSync.existsSync(tempVideoPath)) {
      await fs.unlink(tempVideoPath);
      console.log("Temporary input file cleaned up.");
    }
  }
}
