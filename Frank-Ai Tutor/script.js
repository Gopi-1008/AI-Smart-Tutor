function extractTopicFromMessage(message) {
    const stopWords = new Set([
        "what", "is", "the", "a", "an", "to", "in", "of", "and", "for",
        "how", "why", "explain", "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "our", "their", "his", "her", "me", "us", "them", "do", "does"
    ]);

    const words = message
        .toLowerCase()
        .split(/\W+/)
        .filter(word => word.length > 2 && !stopWords.has(word));

    const firstGood = words.find(word => /^[a-zA-Z]+$/.test(word));

    return firstGood ? firstGood.charAt(0).toUpperCase() + firstGood.slice(1) : "General";
}

// Global variables
let chatHistory = [];
let followUpHistory = [];
let chatHistoryData = {};


//bubble animation
function createBubbles(count) {
    const sidebar = document.querySelector('.sidebar');
    
    for (let i = 0; i < count; i++) {
        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        
        // Random size between 5 and 20px
        const size = Math.floor(Math.random() * 15) + 5;
        bubble.style.width = `${size}px`;
        bubble.style.height = `${size}px`;
        
        // Random position
        const left = Math.floor(Math.random() * 90) + 5;
        bubble.style.left = `${left}%`;
        
        // Random delay and duration
        const delay = Math.random() * 10;
        const duration = Math.random() * 10 + 10;
        bubble.style.animationDelay = `${delay}s`;
        bubble.style.animationDuration = `${duration}s`;
        
        // Random color
        const colorNum = Math.floor(Math.random() * 2) + 1;
        bubble.style.backgroundColor = `var(--bubble-color-${colorNum})`;
        
        sidebar.appendChild(bubble);
    }
}

// Execute when page loads
window.addEventListener('DOMContentLoaded', () => {
    createBubbles(15); // Create 15 additional random bubbles
});

// Toggle sidebar on mobile
document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
        });
    }
});



// Chat functionality
async function callChatbot(message, history = [], isFollowUp = false, currentTopic = '') {
    try {
        const response = await fetch('http://localhost:8000/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: isFollowUp ? 
                    `[Follow-up about ${currentTopic}] ${message}` : 
                    message,
                history: history,
                model: "Llama3-8b-8192",
                current_topic: currentTopic,
                is_follow_up: isFollowUp
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.response;
    } catch (error) {
        console.error('Error calling chatbot:', error);
        throw error;
    }
}

function addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    // Use SVG instead of Font Awesome icon
    avatar.innerHTML = `<svg class="avatar-svg">
        <use href="#${type === 'user' ? 'user-avatar' : 'bot-avatar'}"/>
    </svg>`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = `<p>${content}</p>`;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function addFollowUpMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    // Use SVG instead of Font Awesome icon
    avatar.innerHTML = `<svg class="avatar-svg">
        <use href="#${type === 'user' ? 'user-avatar' : 'bot-avatar'}"/>
    </svg>`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = `<p>${content}</p>`;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    const followUpMessages = document.getElementById('followUpMessages');
    followUpMessages.appendChild(messageDiv);
    followUpMessages.scrollTop = followUpMessages.scrollHeight;
    
    return messageDiv;
}

// Event listener
window.speechSynthesis.cancel();
let unlocked = false;
const welcomeMessage = "Hey there! I'm Frank your AI assistant. How can I help you learn today?";

function enableAndSpeak() {
    if (unlocked) return;
    unlocked = true;

    // This blank utterance on click "unlocks" TTS in Chrome
    speechSynthesis.speak(new SpeechSynthesisUtterance(''));
    
    // Now speak the real welcome
    speak(welcomeMessage);

    // Clean up
    document.removeEventListener('click', enableAndSpeak);
}

document.addEventListener('DOMContentLoaded', () => {
    // Listen for the first click anywhere
    document.addEventListener('click', enableAndSpeak);

    // Still show the welcome text immediately
    addMessage('bot', welcomeMessage);

    // (rest of your DOMContentLoaded setup…)
    setupEventListeners();
    setupClearHistoryButton();
    // Add initial welcome message to follow-up chat
    const followUpMessages = document.getElementById('followUpMessages');
    if (followUpMessages) {
        // Clear any existing messages
        followUpMessages.innerHTML = '';
        // Add the welcome message
        addFollowUpMessage('bot', "Hello! If you have any query related to that topic, feel free to ask me!?");
    }

    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
        try {
            chatHistoryData = JSON.parse(savedHistory);
            updateChatHistoryUI();
        } catch (error) {
            console.error('Error loading chat history:', error);
            chatHistoryData = {};
            localStorage.removeItem('chatHistory');
        }
    }

    setupChatHistorySearch();
});

function setupEventListeners() {
    // Menu toggle
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
    });

    // Chat controls
    setupChatControls();
    
    // Topic selection
    setupTopicSelection();

    // Add follow-up chat controls
    const followUpInput = document.getElementById('followUpInput');
    const followUpSend = document.getElementById('followUpSend');
    
    followUpSend.addEventListener('click', handleFollowUpQuestion);
    followUpInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleFollowUpQuestion();
        }
    });
    
        // Trigger the file input when the upload icon is clicked
        document.getElementById("upload-icon").addEventListener("click", function() {
            document.getElementById("file-upload").click(); // Trigger the hidden file input
        });
    
}

function setupChatControls() {
    const chatInput = document.querySelector('#mainInput');
    const sendButton = document.querySelector('.chat-input button');
    
    if (chatInput && sendButton) {
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
}

function setupTopicSelection() {
    const topicItems = document.querySelectorAll('.topic-list li');
    topicItems.forEach(item => {
        item.addEventListener('click', () => {
            topicItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            const topicName = item.textContent.trim();
            const vizTitle = document.querySelector('.visualization-title');
            if (vizTitle) {
                vizTitle.textContent = `${topicName} - Interactive Model`;
            }
            
            document.getElementById('chatMessages').innerHTML = '';
            addMessage('bot', `I've loaded the ${topicName}. What would you like to know about it?`);
            
            const followUpMessages = document.getElementById('followUpMessages');
            followUpMessages.innerHTML = '';
            addFollowUpMessage('bot', `Hey there! Do you have any questions about ${topicName}? Feel free to ask!`);
            
            chatHistory = [];
            followUpHistory = [];
        });
    });
}

// Fixed formatDate function
function formatDate(dateStr) {
    if (!dateStr) return 'Unknown Date';
    
    try {
        const date = new Date(dateStr);
        if (isNaN(date.getTime())) {
            // If it's not a valid date string that can be parsed by Date constructor
            return dateStr; // Just return the string as is
        }
        
        const today = new Date().toLocaleDateString();
        const yesterday = new Date(Date.now() - 86400000).toLocaleDateString();
        const formattedDate = date.toLocaleDateString();
        
        if (formattedDate === today) return 'Today';
        if (formattedDate === yesterday) return 'Yesterday';
        return formattedDate;
    } catch (error) {
        console.error('Error formatting date:', error);
        return dateStr;
    }
}
function saveChatHistory(topic, question, answer, isFollowUp = false, imageUrl = null) {
    // Skip saving if this is a follow-up message
    if (isFollowUp) return;
    
    try {
        const now = new Date();
        const id = Date.now().toString();
        
        if (!topic || typeof topic !== 'string') {
            topic = 'General Topic';
        }
        
        if (!chatHistoryData[topic]) {
            chatHistoryData[topic] = {
                main: [],
                followUp: []
            };
        }

        const entry = {
            id,
            question,
            answer,
            timestamp: now.toISOString(),
            date: now.toLocaleDateString(),
            time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            isFollowUp: false,
            imageUrl: imageUrl // Store the image URL if available
        };

        // Only add to main chat history
        chatHistoryData[topic].main.push(entry);

        localStorage.setItem('chatHistory', JSON.stringify(chatHistoryData));
        updateChatHistoryUI();
    } catch (error) {
        console.error('Error saving chat history:', error);
    }
}
function setupChatHistorySearch() {
    const searchInput = document.getElementById('historySearch');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            const historyItems = document.querySelectorAll('.history-item');
            
            historyItems.forEach(item => {
                const question = item.querySelector('.history-preview').textContent.toLowerCase();
                const topic = item.dataset.topic.toLowerCase();
                const isVisible = question.includes(searchTerm) || topic.includes(searchTerm);
                item.style.display = isVisible ? 'block' : 'none';
            });
        });
    }
}

function updateChatHistoryUI() {
    const container = document.getElementById('chatHistoryContainer');
    if (!container) return;

    container.innerHTML = '';

    // Group by date
    const groupedHistory = {};

    Object.entries(chatHistoryData).forEach(([topic, messages]) => {
        if (!messages || (!messages.main && !messages.followUp)) return;

        const allMessages = [
            ...(Array.isArray(messages.main) ? messages.main : []), 
            ...(Array.isArray(messages.followUp) ? messages.followUp : [])
        ];

        allMessages.forEach(msg => {
            if (!msg || !msg.date) return; // Skip invalid entries

            if (!groupedHistory[msg.date]) {
                groupedHistory[msg.date] = [];
            }
            groupedHistory[msg.date].push({ ...msg, topic });
        });
    });

    // Sort dates in reverse chronological order
    const sortedEntries = Object.entries(groupedHistory)
        .sort(([dateA], [dateB]) => {
            const dateObjA = new Date(dateA);
            const dateObjB = new Date(dateB);

            if (!isNaN(dateObjA) && !isNaN(dateObjB)) {
                return dateObjB - dateObjA;
            }

            return dateB.localeCompare(dateA);
        });

    // Reverse the order of the messages, so the latest messages appear at the bottom
    sortedEntries.reverse().forEach(([date, messages]) => {
        const group = document.createElement('div');
        group.className = 'history-group';

        const dateDisplay = formatDate(date);

        group.innerHTML = `
            <div class="history-date">${dateDisplay}</div>
            ${messages.map(msg => `
                <div class="history-item" data-topic="${msg.topic}" data-id="${msg.id}">
                    <div class="history-topic">
                        <i class="fas fa-comments"></i>
                        ${msg.topic}
                    </div>
                    <div class="history-preview">${msg.question}</div>
                    <div class="history-time">${msg.time || ''}</div>
                </div>
            `).join('')}
        `;

        container.appendChild(group);
    });

    // Add click handlers
    document.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            loadChatHistory(item.dataset.topic, item.dataset.id);
        });
    });
}


function loadChatHistory(topic, id) {
    if (!chatHistoryData[topic]) return;

    const allMessages = [
        ...(chatHistoryData[topic].main || []),
        ...(chatHistoryData[topic].followUp || [])
    ];
    
    const conversation = allMessages.find(m => m.id === id);
    if (!conversation) return;

    // Clear current chat
    document.getElementById('chatMessages').innerHTML = '';
    
    // Clear current image
    const imageContainer = document.getElementById('generated-image');
    if (imageContainer) {
        imageContainer.innerHTML = '';
    }
    
    // Add the conversation
    addMessage('bot', 'Loading previous conversation...');
    addMessage('user', conversation.question);
    addMessage('bot', conversation.answer);
    
    // If there's an image URL, display it
    if (conversation.imageUrl) {
        if (imageContainer) {
            imageContainer.innerHTML = `
                <img src="${conversation.imageUrl}" alt="Generated image" 
                    style="max-width: 100%; border-radius: 8px; box-shadow: var(--card-shadow);">
            `;
        }
    }

    // Update active state in UI
    document.querySelectorAll('.history-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === id);
    });

    // Update chat history array
    chatHistory = [
        { role: "user", content: conversation.question },
        { role: "assistant", content: conversation.answer }
    ];
}


async function fetchTopic(message) {
    const response = await fetch("http://localhost:8000/extract-topic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: message,
            model: "Llama3-8b-8192"
        })
    });

    const data = await response.json();
    return data.topic || "General";
}

// Fixed sendMessage function
async function sendMessage() {
    const chatInput = document.getElementById('mainInput');
    if (!chatInput) return;
    
    const message = chatInput.value.trim();
    if (!message) return;
    console.log("Message received:", message);
    // Get current topic
    const activeTopicElement = document.querySelector('.topic-list li.active');
    const currentTopic = activeTopicElement ? 
        activeTopicElement.textContent.trim() : 
        '';

    addMessage('user', message);
    chatInput.value = '';

    const loadingMessage = addMessage('bot', '<div class="loading-dots"><span></span><span></span><span></span></div>');

    try {
        // Check if message contains image generation keywords
        const imageKeywords = [
            'generate image','generate the image ','generate the image of ', 'generate an image', 'generate a image', 'generate a image of',
            'create image', 'create an image', 'create a image','create an image of',
            'make image', 'make an image', 'make a image',
            'draw', 'show me', 'show a picture', 'show picture'
          ];
        
          const isImageRequest = imageKeywords.some(keyword => message.toLowerCase().includes(keyword));
          console.log("Is image request detected:", isImageRequest);
        if (isImageRequest) {
            // Extract the image prompt (remove the command words)
            let imagePrompt = message;
            imageKeywords.forEach(keyword => {
                imagePrompt = imagePrompt.toLowerCase().replace(keyword, '').trim();
            });

            // Generate image
            try {
                const imageResponse = await fetch('http://127.0.0.1:8000/generate-image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt: imagePrompt })
                });

                const imageData = await imageResponse.json();
                if (imageResponse.ok) {
                    // Update the image container
                    const imageContainer = document.getElementById('generated-image');
                    if (imageContainer) {
                        imageContainer.innerHTML = `
                            <img src="${imageData.url}" alt="Generated image: ${imagePrompt}" 
                                style="max-width: 100%; border-radius: 8px; box-shadow: var(--card-shadow);">
                        `;
                    }
                    loadingMessage.remove();
                    const response = `I've generated an image based on your prompt: "${imagePrompt}"`;
                    addMessage('bot', response);
                    
                    // Save to chat history with image URL
                    saveChatHistory(currentTopic, message, response, false, imageData.url);
                    
                    // Update chat history in memory
                    chatHistory.push(
                        { role: "user", content: message },
                        { role: "assistant", content: response }
                    );
                } else {
                    throw new Error(imageData.error || 'Failed to generate image');
                }
            } catch (imageError) {
                loadingMessage.remove();
                const errorMsg = `Sorry, I couldn't generate the image. Error: ${imageError.message}`;
                addMessage('bot', errorMsg);
                console.error('Image generation error:', imageError);
            }
        } else {
            // Normal chat response
            const response = await callChatbot(message, chatHistory);
            loadingMessage.remove();
            addMessage('bot', response);
            
            // Try to speak the response
            try {
                speak(response);
            } catch (speechError) {
                console.error('Speech synthesis error:', speechError);
            }

            // Save to chat history
            const extractedTopic = await fetchTopic(message);
            saveChatHistory(extractedTopic, message, response, false);            
            // Update chat history in memory
            chatHistory.push(
                { role: "user", content: message },
                { role: "assistant", content: response }
            );
        }
    } catch (error) {
        console.error('Message sending error:', error);
        loadingMessage.remove();
        addMessage('bot', 'Sorry, I encountered an error. Please try again.');
    }
}

// Updated handleFollowUpQuestion function
async function handleFollowUpQuestion() {
    const followUpInput = document.getElementById('followUpInput');
    if (!followUpInput) return;
    
    const question = followUpInput.value.trim();
    if (!question) return;

    // Add user message
    addFollowUpMessage('user', question);
    followUpInput.value = '';

    // Show loading state
    const loadingMessage = addFollowUpMessage('bot', '<div class="loading-dots"><span></span><span></span><span></span></div>');

    try {
        // Get current topic
        const activeTopicElement = document.querySelector('.topic-list li.active');
        const currentTopic = activeTopicElement ? 
            activeTopicElement.textContent.trim() : 
            'General Topic';

        const response = await fetch('http://localhost:8000/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: question,
                system_prompt: `You name is Frank, an adaptive and slightly witty educational AI assistant designed to help students learn effectively and enjoyably.

🎯 Personality & Style:
- Friendly, supportive, and sometimes funny 😄
- Speak clearly, using simple and engaging language
- Adjust depth and detail based on the user's question and skill level
- Stay focused on the topic — no tangents, distractions, or unnecessary filler
- Make responses fun to read but always informative and structured

🧠 Communication Behavior:
- Use **Markdown formatting** for all responses
- Structure your replies with **bold headers**, bullet points, and whitespace
- Add light emojis where they make the answer more engaging (but don't overdo it)

📘 Response Rules:

1. 📝 **For exam-focused or short-answer questions:**
   - Keep the response under **3 concise sentences**
   - Focus strictly on **key points**
   - Use bullet points *only* when listing standalone facts or steps
   - ❌ Never use bullet points in flowing or incomplete sentences
   - ✅ Use emojis if they add clarity or make the tone friendlier

2. 💡 **For explanation-heavy questions** (e.g., containing "explain", "why", or "how"):
   - Break down complex ideas into **clear, digestible parts**
   - Use **real-world examples**, analogies, or simple metaphors
   - Organize content with **bold section titles** and clean structure
   - Use **numbered steps**, paragraphs, or bullets when needed
   - Keep the flow natural — no robotic language

3. 🖼️ **For image-related prompts:**
   - Generate the image as requested
   - Follow up with a **clear, short explanation**
   - Mention how the image connects to the question or topic
   - Format the description to complement the visual output

🧹 Formatting Guidelines:
- Use **Bold headers** to break sections
- Use *** before subpoints (for Markdown sub-bullets)
- Add 1 blank line between sections for readability
- Do NOT use mid-sentence bullets
- Keep responses **scannable, informative, and visually clean**
- End long answers with a brief **conclusion or summary**

Frank's goal: Teach better, not longer. Make every answer count! ✨
`,
                history: followUpHistory,
                model: "Llama3-8b-8192",
                current_topic: currentTopic
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        loadingMessage.remove();
        addFollowUpMessage('bot', data.response);
        
        // Save to chat history
        saveChatHistory(currentTopic, question, data.response, true);

        // Update follow-up history
        followUpHistory.push(
            { role: "user", content: question },
            { role: "assistant", content: data.response }
        );

    } catch (error) {
        console.error('Follow-up error:', error);
        loadingMessage.remove();
        addFollowUpMessage('bot', 'Sorry, I encountered an error. Please try again.');
    }
}

// Add this function after setupEventListeners
function setupClearHistoryButton() {
    const sidebarTitle = document.querySelector('.chat-history-container');
    if (!sidebarTitle) return;
    
    const clearButton = document.createElement('button');
    clearButton.className = 'clear-history-btn';
    clearButton.innerHTML = '<i class="fas fa-trash"></i> Clear History';
    
    clearButton.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all chat history?')) {
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();
            
            chatHistoryData = {};
            localStorage.removeItem('chatHistory');
            updateChatHistoryUI();
            
            // Clear current chat
            document.getElementById('chatMessages').innerHTML = '';
            addMessage('bot', "Hey there! I'm Frank your AI assistant. How can I help you learn today?");
        }
    });

    // Insert clear button after the sidebar title
    sidebarTitle.parentNode.insertBefore(clearButton, sidebarTitle.nextSibling);
}

// Fixed speak function
function speak(text) {
    try {
        if (!text) return;
        
        if (speechSynthesis.speaking) {
            speechSynthesis.cancel();  // stop any overlapping
        }
        
        if (speechSynthesis.speaking || speechSynthesis.pending) {
            speechSynthesis.cancel();
        }
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1.3;
        utterance.pitch = 1;
        utterance.volume = 1;
        
        // Try to get a preferred voice
        let voices = window.speechSynthesis.getVoices();
        let preferredVoice = voices.find(voice => 
            voice.name.includes('Google') || 
            voice.name.includes('English') || 
            voice.name.includes('US')
        );
        
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
        
        speechSynthesis.speak(utterance);
    } catch (error) {
        console.error('Speech synthesis error:', error);
    }
}

// Speech recognition
let recognition = null;

function startListening(inputId, micButtonId) {
    const input = document.getElementById(inputId);
    const micButton = document.getElementById(micButtonId);
    
    if (speechSynthesis.speaking || speechSynthesis.pending) {
        speechSynthesis.cancel();
    }

    if (!input || !micButton) return;

    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        alert("Speech recognition not supported in this browser.");
        return;
    }

    // Cancel existing recognition
    if (recognition) {
        try {
            recognition.stop();
        } catch (e) {
            console.error("Error stopping speech recognition:", e);
        }
        micButton.classList.remove("listening");
    }

    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.continuous = false;

    micButton.classList.add("listening");

    recognition.onstart = () => {
        console.log("🎤 Listening...");
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
    
        // Replace immediate submission with delayed submission
        setTimeout(() => {
            if (inputId === 'mainInput') {
                sendMessage();
            } else if (inputId === 'followUpInput') {
                handleFollowUpQuestion();
            }
        }, 300); // 300ms delay
    };
        
    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        micButton.classList.remove("listening");
    };

    recognition.onend = () => {
        micButton.classList.remove("listening");
    };

    try {
        recognition.start();
    } catch (error) {
        console.error("Error starting speech recognition:", error);
        micButton.classList.remove("listening");
    }
}

// Trigger the file input when the upload icon is clicked
document.getElementById("upload-icon").addEventListener("click", function() {
    document.getElementById("file-upload").click(); // Trigger the hidden file input
});

// Handle file selection
document.getElementById("file-upload").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        console.log("File uploaded:", file.name);
        // Add further functionality for displaying the file, uploading it to a server, etc.
    }
});
function adjustHeight(textarea) {
    textarea.style.height = 'auto'; // Reset height to auto before recalculating
    textarea.style.height = textarea.scrollHeight + 'px'; // Set height based on content
}

// Add these functions to your script.js file

// Function to toggle sidebar visibility
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContainer = document.querySelector('.main-container');
    
    sidebar.classList.toggle('active');
    
    // Create or remove overlay when sidebar is toggled
    let overlay = document.querySelector('.sidebar-overlay');
    if (!overlay && sidebar.classList.contains('active')) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay active';
        overlay.addEventListener('click', toggleSidebar);
        mainContainer.appendChild(overlay);
    } else if (overlay) {
        overlay.remove();
    }
}

// Event listener for menu toggle button
document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.getElementById('menuToggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', toggleSidebar);
    }
    
    // Adjust chat container height on window resize
    window.addEventListener('resize', function() {
        adjustLayoutForScreenSize();
    });
    
    // Initial adjustment
    adjustLayoutForScreenSize();
    
    // Handle file upload icon click
    const uploadIcon = document.getElementById('upload-icon');
    const fileUpload = document.getElementById('file-upload');
    
    if (uploadIcon && fileUpload) {
        uploadIcon.addEventListener('click', function() {
            fileUpload.click();
        });
    }
    
    // Make inputs submit on Enter key
    const mainInput = document.getElementById('mainInput');
    const followUpInput = document.getElementById('followUpInput');
    
    if (mainInput) {
        mainInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    if (followUpInput) {
        followUpInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                handleFollowUpQuestion();
            }
        });
    }
});

// Function to adjust layout based on screen size
function adjustLayoutForScreenSize() {
    const contentArea = document.querySelector('.content-area');
    const chatContainer = document.querySelector('.chat-container');
    const rightPanel = document.querySelector('.right-panel');
    
    if (window.innerWidth <= 1200) {
        // Stack layout for smaller screens
        if (contentArea) contentArea.style.flexDirection = 'column';
        if (chatContainer) {
            chatContainer.style.width = '100%';
            chatContainer.style.maxWidth = '100%';
        }
        if (rightPanel) {
            rightPanel.style.width = '100%';
            rightPanel.style.maxWidth = '100%';
        }
    } else {
        // Side-by-side layout for larger screens
        if (contentArea) contentArea.style.flexDirection = 'row';
        if (chatContainer) {
            chatContainer.style.width = '60%';
            chatContainer.style.maxWidth = '60%';
        }
        if (rightPanel) {
            rightPanel.style.width = '40%';
            rightPanel.style.maxWidth = '40%';
        }
    }
}

// Add to script.js
function showLoadingState() {
    const imageContainer = document.getElementById('generated-image');
    imageContainer.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner"></div>
            <p>Generating image...</p>
        </div>
    `;
}

// Add to script.js
function handleError(error, type = 'chat') {
    const errorMessages = {
        'chat': 'Sorry, I had trouble processing your message. Please try again.',
        'image': 'Image generation failed. Please try a different prompt.',
        'voice': 'Voice recognition not available. Please type your message.'
    };
    
    addMessage('bot', errorMessages[type]);
    console.error(`${type} Error:`, error);
}


document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        sendMessage();
    }
    // Esc to clear input
    if (e.key === 'Escape') {
        document.getElementById('mainInput').value = '';
    }
});

// Export necessary functions
window.addMessage = addMessage;
window.callChatbot = callChatbot;
window.sendMessage = sendMessage;
window.handleFollowUpQuestion = handleFollowUpQuestion;
window.startListening = startListening;
window.speak = speak;



