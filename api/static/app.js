document.addEventListener('DOMContentLoaded', () => {
    // --- Toggles ---
    const btnChallenge = document.getElementById('btn-challenge');
    const btnTest = document.getElementById('btn-test');
    const viewChallenge = document.getElementById('challenge-view');
    const viewTest = document.getElementById('test-view');

    btnChallenge.addEventListener('click', () => {
        btnChallenge.classList.add('active');
        btnTest.classList.remove('active');
        viewChallenge.classList.add('active');
        viewTest.classList.remove('active');
        viewChallenge.classList.remove('hidden');
        viewTest.classList.add('hidden');
    });

    btnTest.addEventListener('click', () => {
        btnTest.classList.add('active');
        btnChallenge.classList.remove('active');
        viewTest.classList.add('active');
        viewChallenge.classList.remove('active');
        viewTest.classList.remove('hidden');
        viewChallenge.classList.add('hidden');
        loadEvalData();
    });

    // --- Challenge Mode ---
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const sectionsContainer = document.getElementById('results-sections');
    const fusedContainer = document.getElementById('results-fused');
    const visContainer = document.getElementById('results-visual');
    const textContainer = document.getElementById('results-text');
    const loading = document.getElementById('loading');

    const renderCards = (clips, containerElement) => {
        containerElement.innerHTML = '';
        if (!clips || clips.length === 0) {
            containerElement.innerHTML = '<p style="color:var(--text-secondary); text-align:center; width: 100%;">No clips found.</p>';
            return;
        }
        clips.forEach(clip => {
            const card = document.createElement('div');
            card.className = 'video-card';
            
            let videoElement = '';
            if (clip.video_url) {
                const srcUrl = `${clip.video_url}#t=${clip.start_time},${clip.end_time}`;
                videoElement = `
                    <div class="video-player-container">
                        <video controls preload="metadata">
                            <source src="${srcUrl}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                `;
            } else {
                videoElement = `
                    <div class="video-player-container" style="display:flex; align-items:center; justify-content:center; background:#1e293b; color:#94a3b8;">
                        Video File Not Found locally
                    </div>
                `;
            }

            card.innerHTML = `
                ${videoElement}
                <div class="video-info">
                    <div class="video-title">${clip.video_name}</div>
                    <div class="video-time">⏱️ ${clip.start_time.toFixed(1)}s - ${clip.end_time.toFixed(1)}s</div>
                </div>
            `;
            containerElement.appendChild(card);
        });
    };

    const performSearch = async () => {
        const query = searchInput.value.trim();
        if (!query) return;

        // UI Feedback
        sectionsContainer.classList.add('hidden');
        fusedContainer.innerHTML = '';
        visContainer.innerHTML = '';
        textContainer.innerHTML = '';
        loading.classList.remove('hidden');
        searchBtn.disabled = true;

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            const data = await res.json();
            
            loading.classList.add('hidden');
            searchBtn.disabled = false;
            sectionsContainer.classList.remove('hidden');

            renderCards(data.results, fusedContainer);
            renderCards(data.visual, visContainer);
            renderCards(data.text, textContainer);

        } catch (err) {
            console.error(err);
            loading.classList.add('hidden');
            searchBtn.disabled = false;
            sectionsContainer.classList.remove('hidden');
            fusedContainer.innerHTML = '<p style="color:#f87171; text-align:center; width: 100%;">Error occurred during search.</p>';
        }
    };

    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    // --- Test Dashboard ---
    const refreshEvalBtn = document.getElementById('refresh-eval-btn');
    const evalMetrics = document.getElementById('eval-metrics');
    const evalTableBody = document.getElementById('eval-table-body');

    const loadEvalData = async () => {
        refreshEvalBtn.disabled = true;
        refreshEvalBtn.textContent = 'Loading...';
        
        try {
            const res = await fetch('/api/evaluate_results');
            const data = await res.json();
            
            if (data.error) {
                evalMetrics.innerHTML = `<p style="color:#f87171;">${data.error}</p>`;
                evalTableBody.innerHTML = '';
            } else {
                // Metrics
                const metrics = data.metrics;
                const vqaPrec = (metrics.vqa_12frame_precision !== undefined ? metrics.vqa_12frame_precision : 0).toFixed(1);
                evalMetrics.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-value">${metrics.recall_at_1.toFixed(1)}%</div>
                        <div class="metric-label">Recall @ 1</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.recall_at_5.toFixed(1)}%</div>
                        <div class="metric-label">Recall @ 5</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.recall_at_10.toFixed(1)}%</div>
                        <div class="metric-label">Recall @ 10</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${vqaPrec}%</div>
                        <div class="metric-label">VQA 12-Frame Prec.</div>
                    </div>
                    <div class="metric-card" style="border-color: rgba(255,255,255,0.05)">
                        <div class="metric-value" style="color: var(--text-primary); font-size: 2rem;">${metrics.total_queries}</div>
                        <div class="metric-label">Total Queries</div>
                    </div>
                `;

                // Table
                evalTableBody.innerHTML = '';
                data.results.forEach(res => {
                    const tr = document.createElement('tr');
                    
                    let rankClass = 'error';
                    if (res.match_rank === 1) rankClass = 'success';
                    else if (res.match_rank > 1 && res.match_rank <= 10) rankClass = 'warning';
                    
                    const rankText = res.match_rank !== -1 ? `Rank #${res.match_rank}` : 'Not Found';
                    const vqaBadge = res.vqa_exact_match ? ' <span style="color:#4ade80;font-size:0.75rem;padding:2px 6px;background:rgba(74,222,128,0.1);border-radius:4px;">🎯 VQA</span>' : '';
                    
                    tr.innerHTML = `
                        <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${res.query}">${res.query}${vqaBadge}</td>
                        <td>${res.true_video}</td>
                        <td>${res.true_pts}s</td>
                        <td class="match-rank ${rankClass}">${rankText}</td>
                    `;
                    evalTableBody.appendChild(tr);
                });
            }
        } catch (err) {
            console.error(err);
            evalMetrics.innerHTML = `<p style="color:#f87171;">Failed to load evaluation data.</p>`;
        } finally {
            refreshEvalBtn.disabled = false;
            refreshEvalBtn.textContent = 'Refresh Data';
        }
    };

    refreshEvalBtn.addEventListener('click', loadEvalData);
});
