import React, { useState, useCallback, useMemo } from "react";
import type { ChangeEvent, FormEvent } from "react";

// Base URL path for videos (Next.js serves 'public' folder directly)
const BASE_URL = "/";

// --- TRANSLATION DATA ---
const translations = {
  EN: {
    languageName: "English",
    title: "TWITCH HIGHLIGHT ENGINE",
    uploadLabel: "Upload Game Footage (.mp4)",
    buttonGenerate: "GENERATE HIGHLIGHTS (AI DURATION: 30-90s)",
    buttonProcessing: "INITIATING AI... DO NOT CLOSE",
    processingTitle: "AI Analysis in Progress...",
    step1: "1/2: Uploading video and preparing for analysis...",
    step2: "2/2: Initializing AI & Analyzing video frames...",
    step3: "2/2: Deep analysis in progress. This may take several minutes...",
    complete: "Analysis Complete!",
    failed: "Analysis Failed.",
    errorProtocol: "ERROR PROTOCOL:",
    errorSelectFile: "Please select a video file.",
    errorNoHighlights:
      "Processing finished, but no highlights were generated. Check the server console for Python logs.",
    errorDefault: "Failed to process video. Check console for details.",
    loadingNote:
      "Heads up: Initial AI model loading is the longest phase and can take a few minutes.",
    clipsFound: "CLIPS FOUND",
    clipLabel: "CLIP #",
    download: "DOWNLOAD",
    downloadButton: "DOWNLOAD CLIP",
    // New log message translations (optional, but good practice)
    logTitle: "ANALYSIS LOGS",
  },
  RU: {
    languageName: "Русский",
    title: "ДВИЖОК ХАЙЛАЙТОВ TWITCH",
    uploadLabel: "Загрузить Игровой Материал (.mp4)",
    buttonGenerate: "СОЗДАТЬ ХАЙЛАЙТЫ (ДЛИТЕЛЬНОСТЬ ИИ: 30-90с)",
    buttonProcessing: "ЗАПУСК ИИ... НЕ ЗАКРЫВАЙТЕ",
    processingTitle: "ИИ-Анализ в Процессе...",
    step1: "1/2: Загрузка видео и подготовка к анализу...",
    step2: "2/2: Инициализация ИИ и анализ кадров...",
    step3:
      "2/2: Глубокий анализ в процессе. Это может занять несколько минут...",
    complete: "Анализ Завершен!",
    failed: "Анализ Неудачен.",
    errorProtocol: "ПРОТОКОЛ ОШИБОК:",
    errorSelectFile: "Пожалуйста, выберите видеофайл.",
    errorNoHighlights:
      "Обработка завершена, но хайлайты не созданы. Проверьте логи Python.",
    errorDefault: "Не удалось обработать видео. Проверьте консоль.",
    loadingNote:
      "Внимание: Начальная загрузка ИИ-модели — самый долгий этап и может занять несколько минут.",
    clipsFound: "НАЙДЕНО КЛИПОВ",
    clipLabel: "КЛИП №",
    download: "СКАЧАТЬ",
    downloadButton: "СКАЧАТЬ КЛИП",
    // New log message translations
    logTitle: "ЛОГИ АНАЛИЗА",
  },
};

type LanguageKey = "EN" | "RU";
type TranslationSet = (typeof translations)["EN"];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [highlightUrls, setHighlightUrls] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [progressMessage, setProgressMessage] =
    useState<string>("Ready to analyze");
  const [error, setError] = useState<string | null>(null);

  // New state for detailed logs
  const [logMessages, setLogMessages] = useState<string[]>([]);

  // State for current language - SET TO 'RU' AS DEFAULT
  const [language, setLanguage] = useState<LanguageKey>("RU");
  const T: TranslationSet = translations[language];

  // Helper function to format progress message strings (like "1/2: ...")
  const formatProgress = (key: keyof TranslationSet) => {
    const message = T[key] as string;
    // For progress messages, split the step number from the description
    const [step, desc] = message
      .split(/:(.*)/s)
      .map((s) => s.trim())
      .filter(Boolean);
    return { step, desc: desc || step }; // Ensure desc is not null/empty
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    setHighlightUrls([]);
    setError(null);
    setProgressMessage("Ready to analyze");
    setLogMessages([]); // Clear logs on new file selection
  };

  // Helper to add a log message and scroll to bottom (simulated auto-scroll)
  const addLogMessage = (message: string, delay: number = 0) => {
    // Simulate delayed logging (as a real server process would be asynchronous)
    setTimeout(() => {
      setLogMessages((prev) => {
        const nextLogs = [...prev, message];
        // In a real app, you'd scroll the log container here.
        return nextLogs;
      });
    }, delay);
  };

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!file) {
        setError(T.errorSelectFile);
        return;
      }

      setLoading(true);
      setError(null);
      setHighlightUrls([]);
      setLogMessages([]); // Clear logs before new submission

      const formData = new FormData();
      formData.append("video", file);

      try {
        // Step 1: Uploading
        setProgressMessage(T.step1);
        addLogMessage(
          `[CLIENT] Starting upload of ${file.name} (${(
            file.size /
            1024 /
            1024
          ).toFixed(2)} MB)...`
        );

        // Simulate further logging during the initial upload/prep phase
        addLogMessage(`[SERVER] Preparing environment...`, 1000);
        addLogMessage(
          `[SERVER] Received file. Hashing file for integrity check.`,
          2500
        );

        // Step 2: AI Processing (This is the long, single API call)
        setProgressMessage(T.step2);
        addLogMessage(`[SERVER] Starting AI model initialization.`, 3000);

        const timer = setTimeout(() => {
          setProgressMessage(T.step3);
          addLogMessage(
            `[SERVER] Model loaded. Deep frame analysis started.`,
            15000
          );
        }, 15000);

        // NOTE: This fetch call is designed to hit the backend API route defined in analyze.ts (or similar).
        const response = await fetch("/api/generate", {
          method: "POST",
          body: formData,
        });

        clearTimeout(timer);

        const data = await response.json();

        if (!response.ok) {
          console.error("Backend Error Logs:", data.pythonLogs);
          // Append Python logs to the log display on error
          if (data.pythonLogs && Array.isArray(data.pythonLogs)) {
            data.pythonLogs.forEach((log: string, index: number) => {
              addLogMessage(`[PYTHON LOG] ${log}`, 100 * index);
            });
          }
          throw new Error(data.error || T.errorDefault);
        }

        // Success:
        if (data.highlightUrls && data.highlightUrls.length > 0) {
          // Append final success message
          if (data.pythonLogs && Array.isArray(data.pythonLogs)) {
            data.pythonLogs.forEach((log: string, index: number) => {
              addLogMessage(`[PYTHON LOG] ${log}`, 100 * index);
            });
          }

          setHighlightUrls(data.highlightUrls);
          setProgressMessage(T.complete);
          addLogMessage(
            `[CLIENT] ${T.complete}`,
            data.pythonLogs.length * 100 + 500
          );
        } else {
          setError(T.errorNoHighlights);
          setProgressMessage(T.failed);
          addLogMessage(
            `[CLIENT] ${T.failed}. Reason: No highlights found.`,
            500
          );
        }
      } catch (err: unknown) {
        let message = T.errorDefault;
        if (err instanceof Error) {
          message = err.message;
        }
        console.error(err);
        setError(message);
        setProgressMessage(T.failed);
        addLogMessage(`[CLIENT] Fatal Error: ${message}`, 1000);
      } finally {
        setLoading(false);
      }
    },
    [file, T]
  ); // T dependency ensures the translated messages are used

  const { step: currentStep, desc: currentDesc } = useMemo(() => {
    // Determine which description to show in the loading card
    return formatProgress(
      (Object.keys(T).find(
        (key) => T[key as keyof TranslationSet] === progressMessage
      ) as keyof TranslationSet) || "processingTitle"
    );
  }, [progressMessage, T]);

  return (
    // Sony Animation Aesthetic: High contrast, pop-art colors, sharp lines
    <div
      className="min-h-screen bg-gray-900 text-white font-inter"
      style={{
        background: "#101010", // Deep black background
        minHeight: "100vh",
      }}
    >
      <title>AI Video Highlight Generator</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <style>{`
        /* SONY ANIMATION / SPIDER-VERSE STYLE */
        body { font-family: 'Inter', sans-serif; }
        
        /* Comic Panel / Card Style */
        .sony-card {
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
            border: 4px solid #101010; /* Thick black outline */
            box-shadow: 
                4px 4px 0px #FF00AA, /* Offset magenta shadow */
                8px 8px 0px #00FFFF, /* Further offset cyan shadow */
                0 0 20px rgba(0, 255, 255, 0.2); /* Soft glow */
            border-radius: 6px; /* Slightly rounded corners */
        }
        .sony-card:hover { 
            transform: translate(-2px, -2px); /* Slight shift on hover */
            box-shadow: 
                6px 6px 0px #FF00AA,
                10px 10px 0px #00FFFF,
                0 0 30px rgba(255, 0, 170, 0.5); /* Enhanced hover glow */
        }

        /* Title Style: Bold, graphic outline */
        .sony-title {
            font-family: 'Inter', sans-serif;
            font-weight: 900;
            color: #FFFFFF;
            letter-spacing: 2px;
            text-shadow: 
                -2px -2px 0 #000, 
                2px -2px 0 #000, 
                -2px 2px 0 #000, 
                2px 2px 0 #000, /* Thick black outline */
                4px 4px 0 #FF00AA; /* Magenta drop shadow */
        }
        
        /* Button Style */
        .sony-button {
            transition: all 0.15s ease-in-out;
            border: 3px solid #000;
            box-shadow: 4px 4px 0 #FF00AA;
            border-radius: 4px; /* Sharp corners */
        }
        .sony-button:hover:not(:disabled) {
            transform: translate(-1px, -1px);
            box-shadow: 5px 5px 0 #00FFFF; /* Shadow shift on hover */
        }
        .sony-button:disabled {
            box-shadow: none;
            transform: none;
        }

        /* PROGRESS BAR ANIMATION (Unchanged) */
        @keyframes pulse-progress {
          0% { transform: translateX(-100%) scaleX(0); }
          50% { transform: translateX(0) scaleX(1); }
          100% { transform: translateX(100%) scaleX(0); }
        }
        .animate-pulse-progress {
          animation: pulse-progress 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
      `}</style>

      <main className="container mx-auto p-4 sm:p-8">
        {/* === SONY STYLE HEADER === */}
        <header className="mb-10 p-4 sm:p-6 bg-[#00FFFF] border-b-8 border-[#FF00AA]">
          <div className="flex flex-col sm:flex-row justify-between items-center max-w-7xl mx-auto">
            {/* Title */}
            <h1 className="text-4xl sm:text-6xl font-extrabold text-center sm:text-left mb-4 sm:mb-0 sony-title">
              {T.title}
            </h1>

            {/* Language Selector */}
            <button
              onClick={() => setLanguage(language === "EN" ? "RU" : "EN")}
              className={`py-2 px-4 font-bold text-lg text-white sony-button
                  ${
                    language === "EN"
                      ? "bg-blue-600 hover:bg-blue-700"
                      : "bg-[#FF00AA] hover:bg-[#CC0088]"
                  }
                `}
            >
              {language === "EN" ? "РУС / RU" : "ENG / EN"}
            </button>
          </div>
        </header>
        {/* ============================== */}

        <div className="max-w-xl mx-auto">
          {/* LOADING CARD (Nezuko Focused - SONY STYLE) */}
          {loading && (
            <div className="bg-[#1e1e1e] p-8 mb-8 flex flex-col items-center text-center space-y-4 sony-card">
              <img
                src="/nezuko-running-mobile-digital-art-o53j2jqvcljywo2f.webp"
                alt="Nezuko Running Loader"
                // Nezuko is larger and centered
                className="w-40 h-40 object-cover border-4 border-[#FF00AA] transform scale-x-[-1]"
                style={{ borderRadius: "50%", boxShadow: "0 0 15px #00FFFF" }}
              />

              <h2
                className="text-3xl font-extrabold text-[#00FFFF] sony-title"
                style={{ textShadow: "2px 2px 0 #000, 4px 4px 0 #FF00AA" }}
              >
                {T.processingTitle}
              </h2>

              <p className="text-xl font-medium text-gray-200">{currentDesc}</p>

              {/* Visual Progress Bar (Indeterminate) */}
              <div className="w-full bg-gray-700 rounded-sm h-4 overflow-hidden mt-3 border border-[#FF00AA]">
                <div
                  className="h-4 bg-gradient-to-r from-[#FF00AA] to-[#00FFFF] rounded-sm animate-pulse-progress"
                  style={{
                    width: "100%",
                    animationDuration: "2s",
                    animationIterationCount: "infinite",
                  }}
                ></div>
              </div>
              <p className="text-sm text-gray-400 mt-2">{T.loadingNote}</p>

              {/* --- NEW LOG SECTION --- */}
              <div
                className="w-full mt-4 p-4 text-left bg-gray-800 rounded-sm border-2 border-[#00FFFF] max-h-40 overflow-y-auto"
                style={{ boxShadow: "4px 4px 0 #FF00AA" }}
              >
                <h3 className="text-sm font-bold text-[#FF00AA] mb-1 uppercase tracking-wider">
                  {T.logTitle}
                </h3>
                {logMessages.map((log, i) => (
                  <p
                    key={i}
                    className="text-xs text-gray-400 font-mono leading-tight whitespace-pre-wrap"
                  >
                    <span className="text-gray-600 mr-2">[{i + 1}]</span>
                    {log}
                  </p>
                ))}
              </div>
              {/* ----------------------- */}
            </div>
          )}

          {/* MAIN FORM CARD - SONY STYLE */}
          <div className="bg-[#1e1e1e] p-6 mb-12 sony-card">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="flex flex-col space-y-3">
                <label
                  htmlFor="video-upload"
                  className="text-xl font-bold text-[#00FFFF]"
                >
                  <span className="block transition-colors duration-200">
                    {T.uploadLabel}
                  </span>
                </label>
                <input
                  type="file"
                  id="video-upload"
                  accept="video/mp4,video/quicktime,video/x-matroska"
                  onChange={handleFileChange}
                  disabled={loading}
                  className="block w-full text-sm sm:text-base text-gray-200
                    file:mr-4 file:py-3 file:px-6
                    file:rounded-sm file:border-0
                    file:text-base file:font-bold
                    file:bg-gradient-to-r file:from-[#FF00AA] file:to-[#00FFFF] file:text-white
                    hover:file:from-[#CC0088] hover:file:to-[#00CCFF] cursor-pointer transition-all
                    disabled:opacity-40 disabled:cursor-not-allowed
                  "
                />
              </div>

              <button
                type="submit"
                disabled={!file || loading}
                className={`w-full py-4 px-4 font-extrabold text-lg text-white sony-button
                  ${
                    !file || loading
                      ? "bg-gray-700 text-gray-400 cursor-not-allowed opacity-50"
                      : "bg-gradient-to-r from-[#FF00AA] to-[#00FFFF] hover:from-[#CC0088] hover:to-[#00CCFF]"
                  }
                `}
              >
                {loading ? T.buttonProcessing : T.buttonGenerate}
              </button>
            </form>

            {error && (
              <div
                className="mt-4 p-4 bg-red-900/70 text-red-300 rounded-sm border-2 border-red-500"
                style={{ boxShadow: "4px 4px 0 #000" }}
              >
                <strong className="text-red-300">{T.errorProtocol}</strong>{" "}
                {error}
              </div>
            )}
          </div>
        </div>

        {highlightUrls.length > 0 && (
          <div className="mt-16">
            <h2
              className="text-3xl font-bold text-center mb-8 sony-title"
              style={{ textShadow: "2px 2px 0 #000, 4px 4px 0 #00FFFF" }}
            >
              {highlightUrls.length} {T.clipsFound}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
              {highlightUrls.map((url, index) => (
                <div
                  key={index}
                  className="bg-[#1e1e1e] overflow-hidden sony-card"
                >
                  <h3 className="text-xl font-bold p-3 text-center bg-[#101010] text-[#00FFFF] border-b-2 border-[#FF00AA]">
                    {T.clipLabel} {index + 1}
                  </h3>
                  <div className="p-4">
                    <video
                      controls
                      // FIX: Prepend the BASE_URL to the video URL for correct path resolution
                      src={`${BASE_URL}${url}`}
                      className="w-full mb-4 aspect-video bg-black border-4 border-[#FF00AA] shadow-xl"
                    >
                      Your browser does not support the video tag.
                    </video>
                    <a
                      // FIX: Prepend the BASE_URL to the download link
                      href={`${BASE_URL}${url}`}
                      download={`highlight_clip_${index + 1}.mp4`}
                      className="block w-full text-center py-2 px-4 font-extrabold text-white sony-button bg-gradient-to-r from-[#FF00AA] to-[#00FFFF] hover:from-[#CC0088] hover:to-[#00CCFF]"
                    >
                      {T.downloadButton}
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
