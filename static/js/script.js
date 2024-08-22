document.addEventListener("DOMContentLoaded", function () {
  // Initialize Material components
  mdc.autoInit();

  // Initialize text field
  const textField = new mdc.textField.MDCTextField(
    document.querySelector(".mdc-text-field")
  );

  // Initialize circular progress
  const circularProgress = new mdc.circularProgress.MDCCircularProgress(
    document.querySelector(".mdc-circular-progress")
  );

  function formatTimestamp(timestamp) {
    if (!timestamp || typeof timestamp !== 'string' || timestamp.length < 15) {
        return 'Invalid Date';
    }
    try {
        const date = new Date(
            timestamp.slice(0, 4),
            timestamp.slice(4, 6) - 1,
            timestamp.slice(6, 8),
            timestamp.slice(9, 11),
            timestamp.slice(11, 13),
            timestamp.slice(13, 15)
        );
        return date.toLocaleString();
    } catch (error) {
        console.error('Error formatting timestamp:', error);
        return 'Invalid Date';
    }
  }

  // Update timestamp display
  function updateTimestamps() {
    document.querySelectorAll('.screenshot-timestamp').forEach(function(element) {
        const timestamp = element.dataset.timestamp;
        if (timestamp) {
            element.textContent = formatTimestamp(timestamp);
        }
    });
  }

  // Call updateTimestamps when the DOM is loaded
  updateTimestamps();

  // Search functionality
  const searchField = new mdc.textField.MDCTextField(document.getElementById('search-field'));
  const tagFilterSelect = new mdc.select.MDCSelect(document.getElementById('tag-filter-select'));

  const searchButton = document.querySelector(".mdc-text-field__icon--trailing");
  const searchInput = document.getElementById("search-input");

  if (searchButton && searchInput) {
    searchButton.addEventListener("click", performSearch);
    searchInput.addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        performSearch();
      }
    });
  } else {
    console.warn('Search elements not found');
  }

  function performSearch() {
    const query = document.getElementById('search-input').value;
    circularProgress.determinate = false;
    circularProgress.open();
    
    $.post('/search', {query: query}, function(data) {
        let resultsHtml = '<div class="mdc-layout-grid__inner">';
        data.forEach(function(screenshot) {
            resultsHtml += `
                <div class="mdc-layout-grid__cell mdc-layout-grid__cell--span-3">
                    <div class="mdc-card screenshot-card" data-screenshot="${screenshot.filename}">
                        <div class="mdc-card__media mdc-card__media--16-9 screenshot-image" style="background-image: url('/static/screenshots/${screenshot.filename}');"></div>
                        <div class="mdc-card__content">
                            <h2 class="mdc-typography--headline6 screenshot-timestamp" data-timestamp="${screenshot.timestamp}">${formatTimestamp(screenshot.timestamp)}</h2>
                            <p class="mdc-typography--body2 screenshot-status ${getStatusClass(screenshot.ocr_status)}">${getStatusText(screenshot.ocr_status)}</p>
                            <div class="screenshot-tags">
                                ${renderTags(screenshot.tags)}
                            </div>
                            <button class="mdc-button edit-tags-button">Edit Tags</button>
                        </div>
                    </div>
                </div>
            `;
        });
        resultsHtml += '</div>';
        document.getElementById('search-results').innerHTML = resultsHtml;
        
        circularProgress.close();
    });
  } 

  function getStatusClass(status) {
    switch (status) {
        case 'Not yet analyzed':
            return 'status-not-analyzed';
        case 'Analyzing':
            return 'status-analyzing';
        case 'Analyzed':
            return 'status-analyzed';
        default:
            return '';
    }
  }

  function getStatusText(status) {
    switch (status) {
        case 'Not yet analyzed':
            return 'Not yet analyzed';
        case 'Analyzing':
            return 'Analyzing';
        case 'Analyzed':
            return 'Analyzed';
        default:
            return '';
    }
  }

  function renderTags(tags) {
    return tags.map(tag => `<span class="tag">${tag}</span>`).join('');
  }

  // Modal functionality
  function openModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = "block";
    // Trigger reflow
    modal.offsetHeight;
    modal.classList.add('show');
  }

  function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('show');
    setTimeout(() => {
        modal.style.display = "none";
    }, 300); // Wait for the animation to finish
  }

  // Close modal when clicking outside
  window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        closeModal(event.target.id);
    }
  }

  // Close buttons
  document.querySelectorAll('.close').forEach(function(closeBtn) {
    closeBtn.onclick = function() {
        closeModal(this.closest('.modal').id);
    }
  });

  // Screenshot modal
  function openScreenshotModal(filename) {
    const modal = document.getElementById('screenshot-modal');
    const modalImg = document.getElementById('screenshot-modal-image');
    modalImg.src = `/static/screenshots/${filename}`;
    modal.style.display = "block";
  }

  document.addEventListener('click', function(e) {
    const screenshotImage = e.target.closest('.screenshot-image');
    if (screenshotImage) {
        const screenshotCard = screenshotImage.closest('.screenshot-card');
        const filename = screenshotCard.dataset.screenshot;
        openScreenshotModal(filename);
    }
  });

  const modal = document.getElementById('screenshot-modal');
  modal.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
  }

  // Config modal
  const configButton = document.getElementById('config-button');
  if (configButton) {
    configButton.addEventListener('click', function(e) {
      e.preventDefault();
      openModal('config-modal');
    });
  } else {
    console.error('Config button not found');
  }

  // Config modal button handlers
  document.getElementById("ocr-all-button")?.addEventListener("click", function(e) {
    e.preventDefault();
    $.post("/ocr-all", function(data) {
        alert(data.message);
    });
  });

  document.getElementById("delete-all-button")?.addEventListener("click", function(e) {
    e.preventDefault();
    if (confirm("Are you sure you want to delete all screenshots?")) {
        $.post("/delete-all", function(data) {
            alert(data.message);
            location.reload();
        });
    }
  });

  document.getElementById("delete-all-and-db-button")?.addEventListener("click", function(e) {
    e.preventDefault();
    if (confirm("Are you sure you want to delete all screenshots and reset the database? This action cannot be undone.")) {
        $.post("/delete-all-and-reset-db", function(data) {
            alert(data.message);
            location.reload();
        });
    }
  });

  document.getElementById("save-interval-button")?.addEventListener("click", function(e) {
    e.preventDefault();
    const interval = document.getElementById("interval-input").value;
    $.post("/set-interval", { interval: interval }, function(data) {
        alert(data.message);
    });
  });

  // Status updates using Server-Sent Events
  const eventSource = new EventSource('/status-updates');
  eventSource.onmessage = function(event) {
    try {
        const data = JSON.parse(event.data);
        if (data.is_new) {
            // Add new screenshot to the timeline
            addNewScreenshot(data);
        } else {
            // Update existing screenshot status
            updateScreenshotStatus(data);
        }
    } catch (error) {
        console.error('Error parsing SSE data:', error);
        console.log('Raw event data:', event.data);
    }
  };

  function addNewScreenshot(data) {
    const screenshotTimeline = document.querySelector('.screenshot-timeline');
    if (!screenshotTimeline) {
        console.warn('Screenshot timeline not found. New screenshot not added:', data);
        return;
    }

    // Check if the screenshot already exists
    if (document.querySelector(`.screenshot-card[data-screenshot="${data.filename}"]`)) {
        console.warn('Screenshot already exists:', data.filename);
        return;
    }

    const tagsHtml = renderTags(data.tags);
    const newScreenshotHtml = `
        <div class="timeline-item">
            <div class="mdc-card screenshot-card" data-screenshot="${data.filename}" data-timestamp="${data.timestamp}">
                <div class="mdc-card__media mdc-card__media--16-9 screenshot-image" style="background-image: url('/static/screenshots/${data.filename}');">
                    <div class="screenshot-overlay">
                        <span class="material-icons">zoom_in</span>
                    </div>
                </div>
                <div class="mdc-card__content">
                    <h2 class="mdc-typography--headline6 screenshot-timestamp">${formatTimestamp(data.timestamp)}</h2>
                    <p class="mdc-typography--body2 screenshot-status ${getStatusClass(data.status)}">${data.status}</p>
                    <div class="screenshot-tags">
                        ${tagsHtml}
                    </div>
                    <button class="mdc-button edit-tags-button">Edit Tags</button>
                </div>
            </div>
        </div>
    `;
    screenshotTimeline.insertAdjacentHTML('afterbegin', newScreenshotHtml);
  }

  function updateScreenshotStatus(data) {
    const card = document.querySelector(`.screenshot-card[data-screenshot="${data.filename}"]`);
    if (card) {
        const statusElement = card.querySelector('.screenshot-status');
        statusElement.textContent = data.status;
        statusElement.className = `mdc-typography--body2 screenshot-status ${getStatusClass(data.status)}`;
        
        const tagsElement = card.querySelector('.screenshot-tags');
        tagsElement.innerHTML = renderTags(data.tags);
    }
  }

  function filterByTag(tag) {
    fetch(`/filter-by-tag/${encodeURIComponent(tag)}`)
        .then(response => response.json())
        .then(screenshots => {
            const screenshotTimeline = document.querySelector('.screenshot-timeline');
            screenshotTimeline.innerHTML = '';
            screenshots.forEach(screenshot => {
                addNewScreenshot({
                    filename: screenshot.filename,
                    timestamp: screenshot.timestamp,
                    status: screenshot.ocr_text ? 'Analyzed' : 'Not yet analyzed',
                    tags: screenshot.tags
                });
            });
        });
  }

  // Populate tag filter
  function populateTagFilter() {
    fetch('/get-all-tags')
        .then(response => response.json())
        .then(tags => {
            const tagFilterList = document.getElementById('tag-filter-list');
            const uniqueTags = [...new Set(tags)]; // Remove duplicates
            
            // Clear existing tags (except the "All" option)
            while (tagFilterList.children.length > 1) {
                tagFilterList.removeChild(tagFilterList.lastChild);
            }
            
            uniqueTags.forEach(tag => {
                const listItem = document.createElement('li');
                listItem.className = 'mdc-list-item';
                listItem.setAttribute('data-value', tag);
                listItem.innerHTML = `
                    <span class="mdc-list-item__ripple"></span>
                    <span class="mdc-list-item__text">${tag}</span>
                `;
                tagFilterList.appendChild(listItem);
            });
            
            // Reinitialize MDC Select to reflect the new options
            const tagFilterSelect = new mdc.select.MDCSelect(document.getElementById('tag-filter-select'));
            tagFilterSelect.layout();

            // Add event listener for tag selection
            tagFilterSelect.listen('MDCSelect:change', () => {
                const selectedTag = tagFilterSelect.value;
                filterScreenshotsByTag(selectedTag);
            });
        })
        .catch(error => {
            console.error('Error fetching tags:', error);
        });
  }

  function filterScreenshotsByTag(tag) {
    const screenshots = document.querySelectorAll('.screenshot-card');
    screenshots.forEach(screenshot => {
        const tags = Array.from(screenshot.querySelectorAll('.tag')).map(tagElement => tagElement.textContent);
        if (tag === '' || tags.includes(tag)) {
            screenshot.style.display = '';
        } else {
            screenshot.style.display = 'none';
        }
    });
  }

  // Call the function to populate tag filter
  populateTagFilter();

  // Initialize MDC components
  const topAppBar = new mdc.topAppBar.MDCTopAppBar(document.querySelector('.mdc-top-app-bar'));

  // Function to open the edit tags modal
  function openEditTagsModal(filename) {
    fetch(`/get_screenshot_info/${filename}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Screenshot info not found');
            }
            return response.json();
        })
        .then(data => {
            const modal = document.getElementById('edit-tags-modal');
            const tagsInput = document.getElementById('tags-input');
            const saveButton = document.getElementById('save-tags-button');
            
            tagsInput.value = Array.isArray(data.tags) ? data.tags.join(', ') : '';
            modal.setAttribute('data-filename', filename);
            
            modal.style.display = 'block';
            tagsInput.focus();
            
            saveButton.onclick = function() {
                const newTags = tagsInput.value.split(',').map(tag => tag.trim()).filter(tag => tag);
                updateTags(filename, newTags);
                modal.style.display = 'none';
            };
        })
        .catch(error => {
            console.error('Error fetching screenshot info:', error);
            alert('Error fetching screenshot info. Please try again.');
        });
  }

  // Function to update tags
  function updateTags(filename, newTags) {
    fetch('/update_tags', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({filename: filename, tags: newTags}),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to update tags');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            const screenshotCard = document.querySelector(`.screenshot-card[data-screenshot="${filename}"]`);
            const tagsContainer = screenshotCard.querySelector('.screenshot-tags');
            tagsContainer.innerHTML = newTags.map(tag => `<span class="tag">${tag}</span>`).join('');
        } else {
            throw new Error(data.message || 'Failed to update tags');
        }
    })
    .catch(error => {
        console.error('Error updating tags:', error);
        alert('Failed to update tags. Please try again.');
    });
  }

  // Add click event listener to screenshot cards
  document.addEventListener('click', function(e) {
    const editTagsButton = e.target.closest('.edit-tags-button');
    if (editTagsButton) {
        const screenshotCard = editTagsButton.closest('.screenshot-card');
        const filename = screenshotCard.dataset.screenshot;
        openEditTagsModal(filename);
    }
  });

  // Close modal when clicking outside
  window.onclick = function(event) {
    const modal = document.getElementById('edit-tags-modal');
    if (event.target == modal) {
        modal.style.display = "none";
    }
  }
}); 