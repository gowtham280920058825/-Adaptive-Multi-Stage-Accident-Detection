// AMAVE Forensic Evidence Timeline Controller (frontend/script.js)

document.addEventListener("DOMContentLoaded", () => {
    // ---------------------------------------------------------
    // 1. STATE MANAGEMENT
    // ---------------------------------------------------------
    let selectedFile = null;
    let currentReport = null;
    let activeStageNum = 1;
    let activeMode = "live"; // "live" | "research"
    let activeDeviceLabel = "CPU (Fallback)";

    // Mode Toggle Elements
    const btnModeLive = document.getElementById("btnModeLive");
    const btnModeResearch = document.getElementById("btnModeResearch");

    // Header Badges
    const valGpuName = document.getElementById("valGpuName");
    const valServerStatus = document.getElementById("valServerStatus");

    // Stage 01 Elements
    const inputVideoFile = document.getElementById("inputVideoFile");
    const btnSelectFile = document.getElementById("btnSelectFile");
    const lineFileChosen = document.getElementById("lineFileChosen");
    const lblSelectedFileName = document.getElementById("lblSelectedFileName");
    const btnResetFile = document.getElementById("btnResetFile");
    const btnExecuteAnalysis = document.getElementById("btnExecuteAnalysis");

    // Stage 01 Info Displays
    const v1FileName = document.getElementById("v1FileName");
    const v1FileSize = document.getElementById("v1FileSize");
    const v1Resolution = document.getElementById("v1Resolution");
    const v1Duration = document.getElementById("v1Duration");

    // Center Canvas Elements
    const boxInitialIngestion = document.getElementById("boxInitialIngestion");
    const boxActiveMedia = document.getElementById("boxActiveMedia");
    const canvasVideoPlayer = document.getElementById("canvasVideoPlayer");
    const canvasFrameImage = document.getElementById("canvasFrameImage");
    const osdCandidateBadge = document.getElementById("osdCandidateBadge");
    const osdAmaveBadge = document.getElementById("osdAmaveBadge");
    const osdMetaLine = document.getElementById("osdMetaLine");
    const lblCanvasMeta = document.getElementById("lblCanvasMeta");
    const trackFrameThumbnails = document.getElementById("trackFrameThumbnails");
    const lblFrameStripCount = document.getElementById("lblFrameStripCount");

    // Stage 02 Detection
    const v2CandidateCount = document.getElementById("v2CandidateCount");
    const v2MaxConf = document.getElementById("v2MaxConf");
    const txtDetResultTitle = document.getElementById("txtDetResultTitle");

    // Stage 03 AMAVE
    const txtAmaveResultTitle = document.getElementById("txtAmaveResultTitle");
    const txtAmaveFusedScore = document.getElementById("txtAmaveFusedScore");
    const fillCdet = document.getElementById("fillCdet");
    const valCdet = document.getElementById("valCdet");
    const fillSpers = document.getElementById("fillSpers");
    const valSpers = document.getElementById("valSpers");
    const fillSmotion = document.getElementById("fillSmotion");
    const valSmotion = document.getElementById("valSmotion");
    const fillSinter = document.getElementById("fillSinter");
    const valSinter = document.getElementById("valSinter");
    const txtAmaveReasoning = document.getElementById("txtAmaveReasoning");

    // Stage 04 SASA
    const txtSasaSeverity = document.getElementById("txtSasaSeverity");
    const txtSasaScore = document.getElementById("txtSasaScore");
    const listSasaFeatures = document.getElementById("listSasaFeatures");

    // Stage 05 SEDE
    const txtSedePriority = document.getElementById("txtSedePriority");
    const txtSedeAction = document.getElementById("txtSedeAction");
    const listSedeSteps = document.getElementById("listSedeSteps");

    // Stage 06 Final Conclusion
    const fDetVal = document.getElementById("fDetVal");
    const fAmaveVal = document.getElementById("fAmaveVal");
    const fSasaVal = document.getElementById("fSasaVal");
    const fSedeVal = document.getElementById("fSedeVal");
    const fEventsVal = document.getElementById("fEventsVal");
    const fProcTimeVal = document.getElementById("fProcTimeVal");
    const btnReplayAnnotated = document.getElementById("btnReplayAnnotated");
    const btnExportReportJson = document.getElementById("btnExportReportJson");

    // Mini Console Log
    const logOutputList = document.getElementById("logOutputList");

    // Initialize System Health
    fetchHealthStatus();

    // ---------------------------------------------------------
    // 2. TIMELINE NODE INTERACTION CONTROLLER
    // ---------------------------------------------------------
    for (let i = 1; i <= 6; i++) {
        const nodeEl = document.getElementById(`node-stage-${i}`);
        if (nodeEl) {
            nodeEl.addEventListener("click", () => {
                if (nodeEl.classList.contains("locked")) return;
                switchActiveStage(i);
            });
        }
    }

    function switchActiveStage(stageNum) {
        if (activeMode !== "live") {
            switchMode("live");
        }

        activeStageNum = stageNum;

        // Update Timeline Sidebar Node Highlighting
        for (let i = 1; i <= 6; i++) {
            const nodeEl = document.getElementById(`node-stage-${i}`);
            if (nodeEl) {
                if (i === stageNum) {
                    nodeEl.classList.add("active");
                } else {
                    nodeEl.classList.remove("active");
                }
            }
        }

        // Update Right Analysis Panel Stage View
        for (let i = 1; i <= 6; i++) {
            const viewEl = document.getElementById(`view-stage-${i}`);
            if (viewEl) {
                if (i === stageNum) {
                    viewEl.classList.remove("hidden-view");
                    viewEl.classList.add("active-stage-view");
                } else {
                    viewEl.classList.add("hidden-view");
                    viewEl.classList.remove("active-stage-view");
                }
            }
        }

        const metricsView = document.getElementById("view-research-metrics");
        if (metricsView) metricsView.classList.add("hidden-view");

        // Update Bottom Progress Line
        updateBottomProgressLine(stageNum);
    }

    function updateBottomProgressLine(stageNum) {
        for (let i = 1; i <= 6; i++) {
            const pstep = document.getElementById(`pstep-${i}`);
            const pline = document.getElementById(`pline-${i}`);

            if (pstep) {
                if (i < stageNum) {
                    pstep.className = "line-step completed";
                    pstep.querySelector(".pstep-dot").textContent = "✓";
                } else if (i === stageNum) {
                    pstep.className = "line-step active";
                    pstep.querySelector(".pstep-dot").textContent = "●";
                } else {
                    pstep.className = "line-step";
                    pstep.querySelector(".pstep-dot").textContent = "○";
                }
            }
            if (pline) {
                if (i < stageNum) pline.classList.add("completed");
                else pline.classList.remove("completed");
            }
        }
    }

    // ---------------------------------------------------------
    // 3. MODE TOGGLE CONTROLLER (LIVE ANALYSIS vs RESEARCH VIEW)
    // ---------------------------------------------------------
    btnModeLive.addEventListener("click", () => switchMode("live"));
    btnModeResearch.addEventListener("click", () => switchMode("research"));

    function switchMode(mode) {
        activeMode = mode;
        if (mode === "live") {
            btnModeLive.classList.add("active");
            btnModeResearch.classList.remove("active");
            switchActiveStage(activeStageNum);
        } else {
            btnModeResearch.classList.add("active");
            btnModeLive.classList.remove("active");

            // Hide all live stage views, show research metrics view
            for (let i = 1; i <= 6; i++) {
                const viewEl = document.getElementById(`view-stage-${i}`);
                if (viewEl) viewEl.classList.add("hidden-view");
            }
            const metricsView = document.getElementById("view-research-metrics");
            if (metricsView) {
                metricsView.classList.remove("hidden-view");
                metricsView.classList.add("active-stage-view");
            }
        }
    }

    async function fetchHealthStatus() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                valServerStatus.textContent = "ONLINE";
                if (data.gpu && data.gpu.device) {
                    activeDeviceLabel = data.gpu.device;
                    valGpuName.textContent = activeDeviceLabel;
                    lblCanvasMeta.textContent = `FRAME: -- │ FPS: 15.0 │ DEVICE: ${activeDeviceLabel}`;
                }
            }
        } catch (e) {
            valServerStatus.textContent = "OFFLINE";
        }
    }

    // ---------------------------------------------------------
    // 4. FILE INGESTION & METADATA HANDLERS
    // ---------------------------------------------------------
    btnSelectFile.addEventListener("click", () => inputVideoFile.click());
    inputVideoFile.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileIngestion(e.target.files[0]);
        }
    });

    btnResetFile.addEventListener("click", () => {
        selectedFile = null;
        lineFileChosen.classList.add("hidden");
        btnSelectFile.classList.remove("hidden");
        btnExecuteAnalysis.disabled = true;
        inputVideoFile.value = "";
        if (canvasFrameImage) {
            canvasFrameImage.src = "";
            canvasFrameImage.classList.add("hidden");
        }
    });

    function handleFileIngestion(file) {
        selectedFile = file;
        btnSelectFile.classList.add("hidden");
        lineFileChosen.classList.remove("hidden");
        lblSelectedFileName.textContent = file.name;

        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        v1FileName.textContent = file.name;
        v1FileSize.textContent = `${sizeMb} MB`;
        v1Resolution.textContent = "1920x1080 (Estimated)";
        v1Duration.textContent = "~ sec";

        // Load center player preview
        const fileUrl = URL.createObjectURL(file);
        canvasVideoPlayer.src = fileUrl;
        canvasVideoPlayer.load();

        if (canvasFrameImage) canvasFrameImage.classList.add("hidden");
        if (canvasVideoPlayer) canvasVideoPlayer.classList.remove("hidden");

        canvasVideoPlayer.onloadedmetadata = () => {
            if (canvasVideoPlayer.duration && !isNaN(canvasVideoPlayer.duration)) {
                v1Duration.textContent = `${canvasVideoPlayer.duration.toFixed(2)} sec`;
            }
            if (canvasVideoPlayer.videoWidth && canvasVideoPlayer.videoHeight) {
                v1Resolution.textContent = `${canvasVideoPlayer.videoWidth}x${canvasVideoPlayer.videoHeight}`;
            }
        };

        const v1Status = document.getElementById("v1Status");
        const v1FrameCount = document.getElementById("v1FrameCount");
        if (v1Status) v1Status.textContent = "INGESTED (READY FOR ANALYSIS)";
        if (v1FrameCount) v1FrameCount.textContent = "Pending Stride Extraction";

        boxInitialIngestion.classList.add("hidden");
        boxActiveMedia.classList.remove("hidden");

        btnExecuteAnalysis.disabled = false;
        logConsole(`Video ingested: ${file.name} (${sizeMb} MB)`);

        // Unlock Node 1 & Node 2
        setNodeState(1, "completed", "Ingested");
        setNodeState(2, "unlocked", "Ready");
    }

    // ---------------------------------------------------------
    // 5. ANALYSIS PIPELINE EXECUTION CONTROLLER
    // ---------------------------------------------------------
    btnExecuteAnalysis.addEventListener("click", async () => {
        if (!selectedFile) return;

        btnExecuteAnalysis.disabled = true;
        btnExecuteAnalysis.textContent = "EXECUTING FORENSIC PIPELINE...";

        // Step to Stage 02
        setNodeState(2, "active", "Scanning...");
        switchActiveStage(2);
        logConsole("Executing YOLO11-AMF candidate detection scan...");

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data = await res.json();
            currentReport = data;

            await processPipelineResults(data);

        } catch (err) {
            alert(`Forensic Pipeline Error: ${err.message}`);
            logConsole(`ERROR: ${err.message}`);
        } finally {
            btnExecuteAnalysis.disabled = false;
            btnExecuteAnalysis.textContent = "INITIALIZE FORENSIC INVESTIGATION →";
        }
    });

    async function processPipelineResults(data) {
        // Immediately display the annotated evidence frame in canvas
        if (data.annotated_frame_url && canvasFrameImage) {
            canvasFrameImage.src = data.annotated_frame_url;
            canvasFrameImage.classList.remove("hidden");
            if (canvasVideoPlayer) canvasVideoPlayer.classList.add("hidden");
            boxInitialIngestion.classList.add("hidden");
            boxActiveMedia.classList.remove("hidden");
        }

        // Update Stage 01 with extracted metadata
        const v1FrameCount = document.getElementById("v1FrameCount");
        const v1Status = document.getElementById("v1Status");
        if (v1FrameCount && data.total_frames_analyzed) {
            v1FrameCount.textContent = `${data.total_frames_analyzed} Frames (Stride = 2)`;
        }
        if (v1Duration && data.video_duration_sec) {
            v1Duration.textContent = `${data.video_duration_sec} sec`;
        }
        if (v1Status) v1Status.textContent = "PREPROCESSED & ANALYZED";

        // --- STAGE 02: DETECTION ---
        const proposalsCount = (data.candidate_proposals_count !== undefined) 
            ? data.candidate_proposals_count 
            : (data.accident_detected ? 1 : 0);
        v2CandidateCount.textContent = proposalsCount;
        v2MaxConf.textContent = data.detection_confidence_pct || "0.0%";

        if (proposalsCount > 0) {
            txtDetResultTitle.textContent = `${proposalsCount} CANDIDATE PROPOSAL${proposalsCount > 1 ? 'S' : ''} DETECTED`;
            txtDetResultTitle.style.color = "var(--accent-amber)";
            osdCandidateBadge.textContent = `PROPOSALS: ${proposalsCount} (PEAK ${data.detection_confidence_pct})`;
            osdCandidateBadge.style.color = "var(--accent-amber)";
        } else {
            txtDetResultTitle.textContent = "NO CANDIDATE PROPOSALS";
            txtDetResultTitle.style.color = "var(--text-muted)";
            osdCandidateBadge.textContent = "PROPOSALS: 0 (NONE)";
            osdCandidateBadge.style.color = "var(--text-muted)";
        }

        await sleep(800);
        setNodeState(2, "completed", `Proposals: ${proposalsCount}`);

        // --- STAGE 03: AMAVE ---
        setNodeState(3, "active", "Verifying...");
        switchActiveStage(3);
        logConsole("Running AMAVE multi-stage temporal verification...");

        const amaveData = data.amave_verification || {};
        const amaveStatus = amaveData.status || "UNVERIFIED_EVENT";
        const isVerified = (amaveStatus === "VERIFIED_ACCIDENT" || amaveStatus === "VERIFIED");
        const fusedScore = amaveData.score ?? 0.0;

        txtAmaveResultTitle.textContent = isVerified ? "VERIFIED ACCIDENT EVENT" : "UNVERIFIED EVENT (REJECTED)";
        txtAmaveResultTitle.style.color = isVerified ? "var(--accent-green)" : "var(--accent-red)";
        txtAmaveFusedScore.innerHTML = `Fused Verification Score (S<sub>fused</sub>): ${Number(fusedScore).toFixed(4)}`;

        const v3VerifiedWindows = document.getElementById("v3VerifiedWindows");
        if (v3VerifiedWindows) {
            v3VerifiedWindows.textContent = `${data.verified_window_count ?? 0} Window(s) (min 2 required)`;
            v3VerifiedWindows.style.color = isVerified ? "var(--accent-green)" : "var(--accent-amber)";
        }

        osdAmaveBadge.textContent = `AMAVE: ${isVerified ? 'VERIFIED' : 'UNVERIFIED'}`;
        osdAmaveBadge.className = isVerified ? "osd-badge ver" : "osd-badge";

        const stages = amaveData.stages || {};
        updateScoreBar("fillCdet", "valCdet", stages.detection_confidence ?? 0.0);
        updateScoreBar("fillSpers", "valSpers", stages.temporal_persistence ?? 0.0);
        updateScoreBar("fillSmotion", "valSmotion", stages.motion_consistency ?? 0.0);
        updateScoreBar("fillSinter", "valSinter", stages.collision_interaction ?? 0.0);

        txtAmaveReasoning.textContent = isVerified ?
            "Temporal persistence & Farneback optical flow motion confirmed genuine collision kinetics." :
            "Temporal and motion evidence did not provide sufficient support for accident confirmation.";

        await sleep(800);
        setNodeState(3, "completed", isVerified ? "VERIFIED" : "UNVERIFIED");

        if (!isVerified) {
            const v5VerifiedStatus = document.getElementById("v5VerifiedStatus");
            const v5SeverityRating = document.getElementById("v5SeverityRating");
            if (v5VerifiedStatus) {
                v5VerifiedStatus.textContent = "UNVERIFIED_EVENT";
                v5VerifiedStatus.style.color = "var(--accent-red)";
            }
            if (v5SeverityRating) {
                v5SeverityRating.textContent = "N/A (Bypassed)";
                v5SeverityRating.style.color = "var(--text-muted)";
            }

            // BYPASS STAGE 04 & 05 FOR UNVERIFIED EVENTS
            setNodeState(4, "bypassed", "BYPASSED");
            setNodeState(5, "bypassed", "BYPASSED");
            setNodeState(6, "unlocked", "Conclusion");
            switchActiveStage(6);
            renderConclusion(data, false);
            populateFrameStrip(data);
            return;
        }

        // --- STAGE 04: SASA SEVERITY ---
        setNodeState(4, "active", "Assessing...");
        switchActiveStage(4);
        logConsole("Calculating SASA visual scene severity index...");

        const sasaData = data.sasa_severity || {};
        const sasaSev = sasaData.severity || "MODERATE";
        const sasaScore = (sasaData.severity_score !== undefined && sasaData.severity_score !== null) ? sasaData.severity_score : 0.0;

        txtSasaSeverity.textContent = sasaSev;
        txtSasaSeverity.style.color = getSeverityColor(sasaSev);
        txtSasaScore.innerHTML = `Visual Severity Index (S<sub>sasa</sub>): ${Number(sasaScore).toFixed(4)}`;

        listSasaFeatures.innerHTML = "";
        const features = sasaData.explanation || ["Collision visual scene evidence evaluated"];
        features.forEach(feat => {
            const li = document.createElement("li");
            li.textContent = `• ${feat}`;
            listSasaFeatures.appendChild(li);
        });

        await sleep(800);
        setNodeState(4, "completed", sasaSev);

        // --- STAGE 05: SEDE DECISION SUPPORT ---
        setNodeState(5, "active", "Deciding...");
        switchActiveStage(5);
        logConsole("Evaluating SEDE emergency response policy recommendations...");

        const sedeData = data.sede_emergency_decision || {};
        const sedePrio = sedeData.emergency_priority || "MODERATE";
        const sedeAction = sedeData.recommended_action || "Control Room Operator Review Advisory";

        txtSedePriority.textContent = sedePrio;
        txtSedePriority.style.color = getSeverityColor(sedePrio);
        txtSedeAction.textContent = sedeAction;

        const v5VerifiedStatus = document.getElementById("v5VerifiedStatus");
        const v5SeverityRating = document.getElementById("v5SeverityRating");
        if (v5VerifiedStatus) {
            v5VerifiedStatus.textContent = "VERIFIED_ACCIDENT";
            v5VerifiedStatus.style.color = "var(--accent-green)";
        }
        if (v5SeverityRating) {
            v5SeverityRating.textContent = sasaSev;
            v5SeverityRating.style.color = getSeverityColor(sasaSev);
        }

        listSedeSteps.innerHTML = "";
        const steps = sedeData.action_steps || ["Notify control room operator", "Queue video stream for verification"];
        steps.forEach((step, idx) => {
            const li = document.createElement("li");
            li.textContent = `${idx + 1}. ${step}`;
            listSedeSteps.appendChild(li);
        });

        await sleep(800);
        setNodeState(5, "completed", sedePrio);

        // --- STAGE 06: FINAL CONCLUSION ---
        setNodeState(6, "unlocked", "Conclusion");
        switchActiveStage(6);
        renderConclusion(data, true);
        populateFrameStrip(data);
    }

    function renderConclusion(data, isVerified) {
        const fInputVideo = document.getElementById("fInputVideo");
        const fDetStatus = document.getElementById("fDetStatus");
        const fDetConf = document.getElementById("fDetConf");
        const fAmaveStatus = document.getElementById("fAmaveStatus");
        const fVerConf = document.getElementById("fVerConf");
        const fSasaVal = document.getElementById("fSasaVal");
        const fSedeVal = document.getElementById("fSedeVal");
        const fProcTimeVal = document.getElementById("fProcTimeVal");

        if (fInputVideo) {
            fInputVideo.textContent = data.input_video || (selectedFile ? selectedFile.name : "--");
        }

        const pred = data.prediction || (isVerified ? "ACCIDENT DETECTED" : "NO ACCIDENT DETECTED");
        if (fDetStatus) {
            fDetStatus.textContent = pred;
            fDetStatus.style.color = (pred === "ACCIDENT DETECTED") ? "var(--accent-red)" : "var(--accent-green)";
        }

        const fProposalsVal = document.getElementById("fProposalsVal");
        if (fProposalsVal) {
            fProposalsVal.textContent = (data.candidate_proposals_count !== undefined) ? data.candidate_proposals_count : 0;
        }

        if (fDetConf) {
            const conf = data.detection_confidence_pct || ((data.detection_confidence || 0) * 100).toFixed(1) + "%";
            fDetConf.textContent = conf;
        }

        if (fAmaveStatus) {
            const st = data.amave_status || (isVerified ? "VERIFIED" : "UNVERIFIED");
            fAmaveStatus.textContent = st;
            fAmaveStatus.style.color = (st === "VERIFIED" || st === "VERIFIED_ACCIDENT") ? "var(--accent-green)" : "var(--accent-amber)";
        }

        if (fVerConf) {
            const verScore = (data.amave_verification && data.amave_verification.score !== undefined) ?
                (data.amave_verification.score * 100).toFixed(1) + "%" : "--";
            fVerConf.textContent = verScore;
        }

        if (fSasaVal) {
            fSasaVal.textContent = data.severity || (isVerified ? (data.sasa_severity ? data.sasa_severity.severity : "MODERATE") : "N/A");
        }

        if (fSedeVal) {
            fSedeVal.textContent = data.emergency_decision || (isVerified ? (data.sede_emergency_decision ? data.sede_emergency_decision.recommended_action : "Emergency Response Recommendation") : "NO EMERGENCY ACTION (UNVERIFIED)");
        }

        const procTime = data.performance?.total_processing_time_sec ?? data.performance?.processing_time_sec ?? "--";
        if (fProcTimeVal) fProcTimeVal.textContent = `${procTime} sec`;

        // Render Model Performance (Test Set Evaluation Metrics ONLY - Separated from Detection Conf)
        const metrics = data.model_metrics || {};
        // A. Event-Level (Video) Metrics
        const mEventAcc = document.getElementById("mEventAcc");
        const mEventPrec = document.getElementById("mEventPrec");
        const mEventRec = document.getElementById("mEventRec");
        const mEventF1 = document.getElementById("mEventF1");
        const mEventFpr = document.getElementById("mEventFpr");

        // B. YOLO Object Detection Metrics
        const mPrecision = document.getElementById("mPrecision");
        const mRecall = document.getElementById("mRecall");
        const mMap50 = document.getElementById("mMap50");
        const mMap5095 = document.getElementById("mMap5095");
        const mF1 = document.getElementById("mF1");
        const mFpr = document.getElementById("mFpr");
        const mSevAcc = document.getElementById("mSevAcc");

        function formatMetricVal(val, asPercent = true) {
            if (val === undefined || val === null || val === "N/A") return "N/A";
            if (typeof val === "number") {
                return asPercent ? (val <= 1.0 ? (val * 100).toFixed(1) + "%" : val.toFixed(1) + "%") : val.toFixed(3);
            }
            return String(val);
        }

        if (mEventAcc) mEventAcc.textContent = formatMetricVal(metrics.event_level_accuracy);
        if (mEventPrec) mEventPrec.textContent = formatMetricVal(metrics.event_level_precision || metrics.event_precision);
        if (mEventRec) mEventRec.textContent = formatMetricVal(metrics.event_level_recall || metrics.event_recall);
        if (mEventF1) mEventF1.textContent = formatMetricVal(metrics.event_level_f1 || metrics.event_f1);
        if (mEventFpr) mEventFpr.textContent = formatMetricVal(metrics.event_level_fpr || metrics.event_fpr);

        if (mPrecision) mPrecision.textContent = formatMetricVal(metrics.precision);
        if (mRecall) mRecall.textContent = formatMetricVal(metrics.recall);
        if (mMap50) mMap50.textContent = formatMetricVal(metrics.map50);
        if (mMap5095) mMap5095.textContent = formatMetricVal(metrics.map50_95);
        if (mF1) mF1.textContent = formatMetricVal(metrics.f1);
        if (mFpr) mFpr.textContent = formatMetricVal(metrics.false_positive_rate);
        if (mSevAcc) {
            mSevAcc.textContent = metrics.severity_accuracy ?
                formatMetricVal(metrics.severity_accuracy) : "N/A (Validated labels unavailable in dataset)";
        }

        if (data.annotated_frame_url && canvasFrameImage) {
            canvasFrameImage.src = data.annotated_frame_url;
            canvasFrameImage.classList.remove("hidden");
            if (canvasVideoPlayer) canvasVideoPlayer.classList.add("hidden");
            boxInitialIngestion.classList.add("hidden");
            boxActiveMedia.classList.remove("hidden");
        } else if (data.annotated_video_url) {
            canvasVideoPlayer.src = data.annotated_video_url;
            canvasVideoPlayer.load();
            boxInitialIngestion.classList.add("hidden");
            boxActiveMedia.classList.remove("hidden");
        }

        logConsole("Forensic analysis completed. Summary report compiled.");
    }

    btnReplayAnnotated.addEventListener("click", () => {
        if (canvasFrameImage) canvasFrameImage.classList.add("hidden");
        if (canvasVideoPlayer) {
            canvasVideoPlayer.classList.remove("hidden");
            if (canvasVideoPlayer.src) canvasVideoPlayer.play();
        }
    });

    btnExportReportJson.addEventListener("click", () => {
        if (!currentReport) return;
        const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentReport, null, 2));
        const a = document.createElement("a");
        a.href = jsonStr;
        a.download = `amave_forensic_report_${currentReport.report_id || 'export'}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    });

    // ---------------------------------------------------------
    // 6. FRAME STRIP TIMELINE POPULATOR
    // ---------------------------------------------------------
    function populateFrameStrip(data) {
        trackFrameThumbnails.innerHTML = "";
        const totalFrames = data.total_frames_analyzed || 6;
        lblFrameStripCount.textContent = `${totalFrames} FRAMES`;

        for (let f = 1; f <= totalFrames; f++) {
            const frameNumStr = String(f * 25).padStart(4, '0');
            const thumb = document.createElement("div");
            thumb.className = `thumb-node ${f === 1 ? 'active' : ''}`;
            thumb.innerHTML = `<span class="thumb-node-lbl">#${frameNumStr}</span>`;

            thumb.addEventListener("click", () => {
                document.querySelectorAll(".thumb-node").forEach(t => t.classList.remove("active"));
                thumb.classList.add("active");
                osdMetaLine.textContent = `SEQ: #${frameNumStr} │ CONF: ${data.detection_confidence_pct ?? '0.0%'}`;
                lblCanvasMeta.textContent = `FRAME: #${frameNumStr} │ FPS: 15.0 │ DEVICE: ${activeDeviceLabel}`;
            });

            trackFrameThumbnails.appendChild(thumb);
        }
    }

    // ---------------------------------------------------------
    // 7. UTILITY FUNCTIONS
    // ---------------------------------------------------------
    function setNodeState(nodeNum, state, statusText) {
        const nodeEl = document.getElementById(`node-stage-${nodeNum}`);
        const txtEl = document.getElementById(`txtNode${nodeNum}Status`);

        if (!nodeEl) return;

        if (state === "unlocked") {
            nodeEl.classList.remove("locked");
        } else if (state === "active") {
            nodeEl.classList.remove("locked");
            nodeEl.classList.add("active");
        } else if (state === "completed") {
            nodeEl.classList.remove("locked");
            nodeEl.classList.add("completed");
        } else if (state === "bypassed") {
            nodeEl.classList.remove("locked");
            nodeEl.classList.add("bypassed");
        }

        if (txtEl && statusText) txtEl.textContent = statusText;
    }

    function updateScoreBar(fillId, valId, scoreVal) {
        const fillEl = document.getElementById(fillId);
        const valEl = document.getElementById(valId);
        const pct = (scoreVal * 100).toFixed(1);
        if (fillEl) fillEl.style.width = `${pct}%`;
        if (valEl) valEl.textContent = Number(scoreVal).toFixed(4);
    }

    function getSeverityColor(level) {
        switch (level) {
            case "CRITICAL": return "var(--accent-red)";
            case "HIGH": return "var(--accent-amber)";
            case "MODERATE": return "var(--border-highlight)";
            default: return "var(--accent-green)";
        }
    }

    function logConsole(msg) {
        const timeStr = new Date().toTimeString().split(' ')[0];
        const line = document.createElement("div");
        line.className = "log-line";
        line.textContent = `[${timeStr}] ${msg}`;
        logOutputList.appendChild(line);
        logOutputList.scrollTop = logOutputList.scrollHeight;
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
