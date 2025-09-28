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
    // ADJUSTED FIX: Increased the internal Next.js request size limit to 6GB
    // to handle large video files (e.g., 10+ hour streams).
    sizeLimit: "6gb",
  },
};

const PYTHON_SCRIPT_PATH = path.join(process.cwd(), "highlight_generator.py");
const VIDEO_UPLOAD_DIR = path.join(process.cwd(), "public", "temp");
const VIDEO_OUTPUT_DIR = path.join(process.cwd(), "public", "videos");

// FIX: Added NextApiRequest type to 'req' and explicit return type.
const parseForm = (
  req: NextApiRequest
): Promise<{ videoPath: string; originalFilename: string }> => {
  return new Promise((resolve, reject) => {
    const form = new IncomingForm({
      uploadDir: VIDEO_UPLOAD_DIR,
      keepExtensions: true,
      // ADJUSTED: Increased formidable's maxFileSize to 6GB (6 * 1024 * 1024 * 1024 bytes)
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

      // FIX: Used non-null assertion operator (!) because we are confident
      // that if videoFile exists after formidable parse, these fields will be strings.
      resolve({
        videoPath: videoFile.filepath!,
        originalFilename: videoFile.originalFilename!,
      });
    });
  });
};

// FIX: Added NextApiRequest and NextApiResponse types to 'req' and 'res'.
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

  await fs.mkdir(VIDEO_UPLOAD_DIR, { recursive: true });
  await fs.mkdir(VIDEO_OUTPUT_DIR, { recursive: true });

  let tempVideoPath: string | null = null;

  try {
    // FIX: Destructuring is now safe because parseForm has an explicit return type.
    const { videoPath, originalFilename } = await parseForm(req);
    tempVideoPath = videoPath;

    const uniqueId =
      path.parse(path.basename(originalFilename)).name + "_" + Date.now();
    const jobOutputDir = path.join(VIDEO_OUTPUT_DIR, uniqueId);
    await fs.mkdir(jobOutputDir, { recursive: true });

    const pythonProcess = spawn("python", [
      PYTHON_SCRIPT_PATH,
      tempVideoPath,
      jobOutputDir,
    ]);

    let pythonStdout = "";
    let pythonStderr = "";

    pythonProcess.stdout.on("data", (data) => {
      const output = data.toString();
      pythonStdout += output;
      console.log("Python stdout chunk:", output);
    });

    pythonProcess.stderr.on("data", (data) => {
      const output = data.toString();
      pythonStderr += output;
      console.error("Python stderr chunk:", output);
    });

    const exitCode = await new Promise<number>((resolve) => {
      pythonProcess.on("close", resolve);
    });

    if (exitCode !== 0) {
      console.error(`Python script exited with code ${exitCode}`);
      throw new Error(
        `Video processing failed. Check Python logs in the server console.`
      );
    }

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
      const absolutePaths: string[] = JSON.parse(jsonString);

      // FIX: Parameter 'absPath' explicitly typed as string.
      highlightUrls = absolutePaths.map((absPath: string) => {
        const relativePath = path
          .relative(path.join(process.cwd(), "public"), absPath)
          .replace(/\\/g, "/");
        return "/" + relativePath;
      });
    }

    res.status(200).json({
      highlightUrls,
      message: "Processing complete. Highlights generated.",
      pythonLogs: pythonStderr,
    });
  } catch (error: unknown) {
    // FIX: Safely handles 'error' of type 'unknown' and extracts message.
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
    if (tempVideoPath && fsSync.existsSync(tempVideoPath)) {
      await fs.unlink(tempVideoPath);
      console.log("Temporary input file cleaned up.");
    }
  }
}
