/**
 * SentinAI // Frontend Dashboard Controller
 * Handles real-time telemetry, camera/video streaming, file uploads, and forensic incident log.
 */

let currentFilter = 'all';
let allIncidents = [];
let isStreamingActive = false;
let activeModalIncidentId = null;

// DOM Elements
const streamImage = document.getElementById('streamImage');
const btnStartLive = document.getElementById('btnStartLive');
const btnStopStream = document.getElementById('btnStopStream');
const cameraSourceSelect = document.getElementById('cameraSourceSelect');
const btnUploadVideo = document.getElementById('btnUploadVideo');
const videoFileInput = document.getElementById('videoFileInput');
const btnRefreshIncidents = document.getElementById('btnRefreshIncidents');

const systemStatusPill = document.getElementById('systemStatusPill');
const fpsCounter = document.getElementById('fpsCounter');
const activeTracksCounter = document.getElementById('activeTracksCounter');
const threatLevelBadge = document.getElementById('threatLevelBadge');

const hudSourceName = document.getElementById('hudSourceName');
const hudThreatPill = document.getElementById('hudThreatPill');
const hudThreatText = document.getElementById('hudThreatText');
const hudClock = document.getElementById('hudClock');
const statusStripMessage = document.getElementById('statusStripMessage');

const incidentGrid = document.getElementById('incidentGrid');
const incidentCountBadge = document.getElementById('incidentCountBadge');

// Modals
const videoModal = document.getElementById('videoModal');
const modalVideoPlayer = document.getElementById('modalVideoPlayer');
const videoModalTitle = document.getElementById('videoModalTitle');
const videoModalMeta = document.getElementById('videoModalMeta');
const btnDownloadVideo = document.getElementById('btnDownloadVideo');

const imageModal = document.getElementById('imageModal');
const modalImageDisplay = document.getElementById('modalImageDisplay');
const imageModalTitle = document.getElementById('imageModalTitle');
const imageModalMeta = document.getElementById('imageModalMeta');
const btnDownloadImage = document.getElementById('btnDownloadImage');

const reportModal = document.getElementById('reportModal');
const reportSummaryText = document.getElementById('reportSummaryText');
const reportJsonViewer = document.getElementById('reportJsonViewer');
const reportModalMeta = document.getElementById('reportModalMeta');

// ============================================================================
// Initialization & Clocks
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    updateClock();
    setInterval(updateClock, 1000);

    // Initial telemetry poll & incidents fetch
    pollStatus();
    setInterval(pollStatus, 1200);

    loadIncidents();
    // Auto-refresh incidents every 8 seconds
    setInterval(loadIncidents, 8000);

    setupEventListeners();
});

function updateClock() {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    hudClock.textContent = timeStr;
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    btnStartLive.addEventListener('click', () => {
        const selectedSource = cameraSourceSelect.value;
        startStream(selectedSource);
    });

    btnStopStream.addEventListener('click', () => {
        stopStream();
    });

    btnUploadVideo.addEventListener('click', () => {
        videoFileInput.click();
    });

    videoFileInput.addEventListener('change', handleFileUpload);

    btnRefreshIncidents.addEventListener('click', async () => {
        const svgIcon = btnRefreshIncidents.querySelector('svg');
        if (svgIcon) svgIcon.style.animation = 'spin 0.8s linear infinite';
        btnRefreshIncidents.disabled = true;
        statusStripMessage.textContent = "Clearing feed buffers, tracking memory, and threat alarms...";

        try {
            const resp = await fetch('/api/feed/reset', { method: 'POST' });
            const data = await resp.json();

            // Re-fetch forensic log
            await loadIncidents();

            // Refresh live stream image cache
            if (isStreamingActive && streamImage) {
                streamImage.src = `/video_feed?t=${Date.now()}`;
            }

            // Immediately reset threat HUD badges to Normal
            threatLevelBadge.className = 'threat-badge threat-level-0';
            threatLevelBadge.textContent = "NORMAL";
            hudThreatPill.className = "threat-pill threat-pill-safe";
            hudThreatText.textContent = "STATUS: NORMAL";

            statusStripMessage.textContent = data.message || "Feed & memory buffers cleared. Monitoring active.";
        } catch (err) {
            console.error("Error resetting feed:", err);
            statusStripMessage.textContent = "Feed refresh completed.";
            await loadIncidents();
        } finally {
            setTimeout(() => {
                if (svgIcon) svgIcon.style.animation = '';
                btnRefreshIncidents.disabled = false;
            }, 600);
        }
    });

    // Filter tabs
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            currentFilter = e.target.dataset.filter;
            renderIncidents();
        });
    });

    // Keyboard escape to close modals
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeVideoModal();
            closeImageModal();
            closeReportModal();
        }
    });
}

// ============================================================================
// Stream Control APIs
// ============================================================================

async function startStream(source) {
    btnStartLive.disabled = true;
    statusStripMessage.textContent = "Initializing AI stream and loading models...";
    systemStatusPill.textContent = "STARTING...";
    systemStatusPill.className = "status-pill status-analyzing";

    try {
        const response = await fetch('/api/stream/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source })
        });
        const data = await response.json();

        if (data.success) {
            isStreamingActive = true;
            btnStopStream.disabled = false;
            hudSourceName.innerHTML = `<span class="dot"></span> SOURCE: ${data.source.toUpperCase()}`;
            statusStripMessage.textContent = data.message || "Active surveillance running.";
            
            // Reload stream image cache to reconnect immediately
            streamImage.src = `/video_feed?t=${Date.now()}`;
        } else {
            alert("Could not start stream: " + (data.message || "Unknown error"));
            btnStartLive.disabled = false;
        }
    } catch (err) {
        console.error("Start stream error:", err);
        statusStripMessage.textContent = "Failed to connect to backend stream.";
        btnStartLive.disabled = false;
    }
}

async function stopStream() {
    btnStopStream.disabled = true;
    statusStripMessage.textContent = "Stopping stream...";

    try {
        const response = await fetch('/api/stream/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();

        isStreamingActive = false;
        btnStartLive.disabled = false;
        btnStopStream.disabled = true;
        hudSourceName.innerHTML = `<span class="dot"></span> SOURCE: STANDBY`;
        statusStripMessage.textContent = data.message || "Stream stopped. System in standby.";

        // Reload standby image
        streamImage.src = `/video_feed?t=${Date.now()}`;
        loadIncidents();
    } catch (err) {
        console.error("Stop stream error:", err);
        btnStopStream.disabled = false;
    }
}

// ============================================================================
// File Upload Handling
// ============================================================================

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);
    formData.append('auto_start', 'true');

    btnUploadVideo.disabled = true;
    btnUploadVideo.innerHTML = `<span>Uploading ${file.name.substring(0, 15)}...</span>`;
    statusStripMessage.textContent = `Uploading "${file.name}" to neural processing engine...`;

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            statusStripMessage.textContent = `Upload complete. Analyzing video sequence: ${file.name}`;
            isStreamingActive = true;
            btnStartLive.disabled = true;
            btnStopStream.disabled = false;
            hudSourceName.innerHTML = `<span class="dot"></span> SOURCE: FILE [${file.name.toUpperCase()}]`;

            // Refresh video feed stream
            streamImage.src = `/video_feed?t=${Date.now()}`;
        } else {
            alert("Upload failed: " + (data.error || "Unknown error"));
            statusStripMessage.textContent = "Upload failed.";
        }
    } catch (err) {
        console.error("Upload error:", err);
        alert("Upload error occurred while sending video to server.");
    } finally {
        btnUploadVideo.disabled = false;
        btnUploadVideo.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <span>Upload Video File</span>
        `;
        videoFileInput.value = '';
    }
}

// ============================================================================
// Telemetry & Status Polling
// ============================================================================

async function pollStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) return;
        const status = await response.json();

        // Update telemetry elements
        fpsCounter.innerHTML = `${status.fps.toFixed(1)} <span class="unit">FPS</span>`;
        activeTracksCounter.textContent = status.active_tracks;

        // Threat level badge
        const level = status.threat_level;
        threatLevelBadge.className = `threat-badge threat-level-${level}`;
        if (level === 0) {
            threatLevelBadge.textContent = "NORMAL";
            hudThreatPill.className = "threat-pill threat-pill-safe";
            hudThreatText.textContent = "STATUS: NORMAL";
        } else if (level === 1) {
            threatLevelBadge.textContent = "SUSPICIOUS";
            hudThreatPill.className = "threat-pill threat-pill-warning";
            hudThreatText.textContent = `ALERT: ${status.threat_label}`;
        } else {
            threatLevelBadge.textContent = "THREAT CONFIRMED";
            hudThreatPill.className = "threat-pill threat-pill-danger";
            hudThreatText.textContent = `CRITICAL: ${status.threat_label}`;
        }

        // MongoDB Cloud Archive Status Badge
        const mongoBadge = document.getElementById('mongoStatusBadge');
        if (mongoBadge && status.mongodb) {
            const m = status.mongodb;
            if (!m.enabled) {
                mongoBadge.textContent = "LOCAL ONLY";
                mongoBadge.className = "status-pill status-ready";
                mongoBadge.title = "MongoDB cloud archiving disabled in .env (MONGODB_ENABLED=false)";
            } else if (m.connected) {
                const pendingText = m.queue_pending > 0 ? ` (${m.queue_pending} QUEUED)` : "";
                mongoBadge.textContent = `ONLINE${pendingText}`;
                mongoBadge.className = "status-pill status-streaming";
                mongoBadge.title = `Connected to MongoDB database: '${m.database}', collection: '${m.collection}'`;
            } else if (m.last_error) {
                mongoBadge.textContent = "OFFLINE";
                mongoBadge.className = "status-pill status-error";
                mongoBadge.title = `MongoDB Error: ${m.last_error}`;
            } else {
                mongoBadge.textContent = "CONNECTING...";
                mongoBadge.className = "status-pill status-analyzing";
                mongoBadge.title = "Connecting to MongoDB...";
            }
        }

        // Streaming state updates
        if (status.is_streaming) {
            isStreamingActive = true;
            btnStartLive.disabled = true;
            btnStopStream.disabled = false;

            if (status.source_type === 'camera') {
                systemStatusPill.textContent = "LIVE MONITORING";
                systemStatusPill.className = "status-pill status-streaming";
            } else {
                systemStatusPill.textContent = "ANALYZING VIDEO";
                systemStatusPill.className = "status-pill status-analyzing";
            }

            if (status.source) {
                hudSourceName.innerHTML = `<span class="dot"></span> SOURCE: ${status.source.toUpperCase()}`;
            }
        } else {
            // When an uploaded video reaches the end
            if (isStreamingActive && status.status === "COMPLETED") {
                isStreamingActive = false;
                btnStartLive.disabled = false;
                btnStopStream.disabled = true;
                systemStatusPill.textContent = "STANDBY";
                systemStatusPill.className = "status-pill status-ready";
                hudSourceName.innerHTML = `<span class="dot"></span> SOURCE: STANDBY`;
                statusStripMessage.textContent = status.status_message;
                loadIncidents();
            }
        }
    } catch (err) {
        // Silently catch network dropouts
    }
}

// ============================================================================
// Incident Locker Loading & Rendering
// ============================================================================

async function loadIncidents() {
    try {
        const response = await fetch('/api/incidents');
        if (!response.ok) return;
        allIncidents = await response.json();
        incidentCountBadge.textContent = `${allIncidents.length} Recorded Incidents`;
        renderIncidents();
    } catch (err) {
        console.error("Failed to load incidents:", err);
    }
}

function renderIncidents() {
    let filtered = allIncidents;
    if (currentFilter === 'video') {
        filtered = allIncidents.filter(inc => inc.video !== null);
    } else if (currentFilter === 'screenshot') {
        filtered = allIncidents.filter(inc => inc.screenshot !== null);
    }

    if (filtered.length === 0) {
        incidentGrid.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <span>No recorded incidents found matching "${currentFilter}".</span>
            </div>
        `;
        return;
    }

    incidentGrid.innerHTML = filtered.map(inc => {
        const hasVideo = inc.video !== null;
        const hasScreenshot = inc.screenshot !== null;
        const hasReport = inc.report !== null;

        const screenshotSrc = hasScreenshot 
            ? `/api/incidents/screenshot/${inc.screenshot}`
            : '/static/img/video_placeholder.svg';

        const summaryText = inc.report_summary 
            ? inc.report_summary 
            : "Physical altercation detected. Buffered evidence stored in secure forensic archive.";

        return `
            <div class="incident-card" id="card-${inc.id}">
                <div class="card-media-wrapper" onclick="handleMediaClick('${inc.id}')">
                    <img src="${screenshotSrc}" alt="Incident Screenshot" loading="lazy" onerror="this.src='/static/img/video_placeholder.svg'">
                    <div class="media-play-overlay">
                        <div class="play-circle">
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <polygon points="6 3 20 12 6 21 6 3"></polygon>
                            </svg>
                        </div>
                    </div>
                    <div class="card-threat-badge">${inc.threat_label}</div>
                    <div class="card-time-badge">${inc.timestamp}</div>
                </div>

                <div class="card-content">
                    <div class="card-header-line">
                        <span class="incident-id">${inc.id}</span>
                        ${inc.video_size_mb > 0 ? `<span class="incident-size">${inc.video_size_mb} MB</span>` : ''}
                    </div>

                    <p class="incident-summary">${escapeHtml(summaryText)}</p>

                    <div class="card-actions">
                        ${hasScreenshot ? `
                            <button class="btn-card" onclick="openImageModal('${inc.id}', '${inc.screenshot}', '${inc.timestamp}')">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                                    <polyline points="21 15 16 10 5 21"></polyline>
                                </svg>
                                <span>Keyframe</span>
                            </button>
                        ` : ''}

                        ${hasVideo ? `
                            <button class="btn-card highlight" onclick="openVideoModal('${inc.id}', '${inc.video}', '${inc.timestamp}')">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                                </svg>
                                <span>Play Video</span>
                            </button>
                            <a class="btn-card" href="/api/incidents/video/${inc.video}" download title="Download MP4">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="7 10 12 15 17 10"></polyline>
                                    <line x1="12" y1="15" x2="12" y2="3"></line>
                                </svg>
                            </a>
                        ` : ''}

                        ${hasReport ? `
                            <button class="btn-card" onclick="openReportModal('${inc.id}', '${inc.timestamp}')">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                    <polyline points="14 2 14 8 20 8"></polyline>
                                    <line x1="16" y1="13" x2="8" y2="13"></line>
                                    <line x1="16" y1="17" x2="8" y2="17"></line>
                                    <polyline points="10 9 9 9 8 9"></polyline>
                                </svg>
                                <span>Report</span>
                            </button>
                        ` : ''}

                        <button class="btn-card btn-card-delete" onclick="deleteIncident('${inc.id}', event)" title="Delete Incident Evidence">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                            <span>Delete</span>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function handleMediaClick(incidentId) {
    const inc = allIncidents.find(i => i.id === incidentId);
    if (!inc) return;

    if (inc.video) {
        openVideoModal(inc.id, inc.video, inc.timestamp);
    } else if (inc.screenshot) {
        openImageModal(inc.id, inc.screenshot, inc.timestamp);
    }
}

// ============================================================================
// Modal Handlers
// ============================================================================

let currentVideoFilename = '';
let currentPlayerMode = 'native';

function setPlayerMode(mode) {
    currentPlayerMode = mode;
    const btnNative = document.getElementById('btnPlayerModeNative');
    const btnStream = document.getElementById('btnPlayerModeStream');
    const fallbackPlayer = document.getElementById('videoFallbackPlayer');
    const fallbackImg = document.getElementById('modalFallbackStream');

    if (mode === 'stream') {
        if (btnNative) {
            btnNative.classList.remove('active');
            btnNative.style.background = 'transparent';
        }
        if (btnStream) {
            btnStream.classList.add('active');
            btnStream.style.background = 'var(--accent)';
        }
        modalVideoPlayer.pause();
        modalVideoPlayer.style.display = 'none';
        if (fallbackPlayer && fallbackImg && currentVideoFilename) {
            fallbackImg.src = `/api/incidents/stream/${encodeURIComponent(currentVideoFilename)}?t=${Date.now()}`;
            fallbackPlayer.style.display = 'block';
        }
    } else {
        if (btnStream) {
            btnStream.classList.remove('active');
            btnStream.style.background = 'transparent';
        }
        if (btnNative) {
            btnNative.classList.add('active');
            btnNative.style.background = 'var(--accent)';
        }
        if (fallbackImg) fallbackImg.src = '';
        if (fallbackPlayer) fallbackPlayer.style.display = 'none';
        modalVideoPlayer.style.display = 'block';
        if (currentVideoFilename) {
            const videoUrl = `/api/incidents/video/${encodeURIComponent(currentVideoFilename)}`;
            modalVideoPlayer.src = videoUrl;
            modalVideoPlayer.load();
            modalVideoPlayer.play().catch(e => console.debug("Autoplay prevented:", e));
        }
    }
}

function restartFallbackStream() {
    const fallbackImg = document.getElementById('modalFallbackStream');
    if (fallbackImg && currentVideoFilename) {
        fallbackImg.src = `/api/incidents/stream/${encodeURIComponent(currentVideoFilename)}?t=${Date.now()}`;
    }
}

function openVideoModal(id, videoFilename, timestamp) {
    activeModalIncidentId = id;
    currentVideoFilename = videoFilename;
    videoModalTitle.textContent = `Evidence Recording: ${id}`;
    videoModalMeta.textContent = `Recorded: ${timestamp}`;
    
    const videoUrl = `/api/incidents/video/${encodeURIComponent(videoFilename)}`;
    btnDownloadVideo.href = videoUrl;

    const errorBanner = document.getElementById('videoErrorBanner');
    if (errorBanner) errorBanner.style.display = 'none';

    // Open with native mode by default
    setPlayerMode('native');
    videoModal.classList.add('active');
}

function closeVideoModal() {
    activeModalIncidentId = null;
    currentVideoFilename = '';
    videoModal.classList.remove('active');

    try {
        modalVideoPlayer.pause();
        modalVideoPlayer.removeAttribute('src');
        modalVideoPlayer.load();
    } catch (e) {}

    const fallbackImg = document.getElementById('modalFallbackStream');
    if (fallbackImg) fallbackImg.src = '';
    const fallbackPlayer = document.getElementById('videoFallbackPlayer');
    if (fallbackPlayer) fallbackPlayer.style.display = 'none';
    const errorBanner = document.getElementById('videoErrorBanner');
    if (errorBanner) errorBanner.style.display = 'none';
    modalVideoPlayer.style.display = 'block';
}

if (modalVideoPlayer) {
    modalVideoPlayer.addEventListener('error', (e) => {
        console.warn("Native video playback issue detected. Auto-switching to CPU MJPEG stream.", e);
        const errorBanner = document.getElementById('videoErrorBanner');
        if (errorBanner) {
            errorBanner.style.display = 'block';
            setTimeout(() => { errorBanner.style.display = 'none'; }, 4000);
        }
        setPlayerMode('stream');
    });
}

function openImageModal(id, imageFilename, timestamp) {
    activeModalIncidentId = id;
    imageModalTitle.textContent = `Keyframe Evidence: ${id}`;
    imageModalMeta.textContent = `Captured: ${timestamp}`;
    
    const imageUrl = `/api/incidents/screenshot/${imageFilename}`;
    modalImageDisplay.src = imageUrl;
    btnDownloadImage.href = imageUrl;

    imageModal.classList.add('active');
}

function closeImageModal() {
    activeModalIncidentId = null;
    imageModal.classList.remove('active');
    modalImageDisplay.src = '';
}

function openReportModal(id, timestamp) {
    const inc = allIncidents.find(i => i.id === id);
    if (!inc || !inc.report) return;

    reportModalMeta.textContent = `Forensic File: ${id} | ${timestamp}`;
    reportSummaryText.textContent = inc.report_summary || "No textual summary found.";
    reportJsonViewer.textContent = JSON.stringify(inc.report, null, 2);

    reportModal.classList.add('active');
}

function closeReportModal() {
    reportModal.classList.remove('active');
}

async function deleteIncident(id, event) {
    if (event) {
        event.stopPropagation();
    }

    if (!confirm(`Permanently delete incident "${id}" and all associated evidence files?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/incidents/${encodeURIComponent(id)}`, {
            method: 'DELETE'
        });
        const data = await response.json();

        if (data.success) {
            // Animate card removal
            const card = document.getElementById(`card-${id}`);
            if (card) {
                card.style.transition = 'all 0.35s ease';
                card.style.opacity = '0';
                card.style.transform = 'scale(0.85)';
                setTimeout(() => {
                    card.remove();
                }, 350);
            }

            // Remove from array and update counter badge
            allIncidents = allIncidents.filter(i => i.id !== id);
            incidentCountBadge.textContent = `${allIncidents.length} Recorded Incidents`;

            // If empty, show empty state
            if (allIncidents.length === 0) {
                renderIncidents();
            }

            // Close modals if currently displaying this incident
            if (activeModalIncidentId === id) {
                closeVideoModal();
                closeImageModal();
                closeReportModal();
            }
        } else {
            alert('Failed to delete incident: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Delete error:', err);
        alert('Failed to delete incident.');
    }
}

function deleteActiveModalIncident() {
    if (activeModalIncidentId) {
        deleteIncident(activeModalIncidentId);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}
