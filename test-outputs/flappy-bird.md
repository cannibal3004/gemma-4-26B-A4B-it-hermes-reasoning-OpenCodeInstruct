This is a complete, single-file solution. I have combined the HTML, CSS, and JavaScript into one block so you can simply save it as an `.html`_ file (e.g., `flappybird.html`) and open it in any browser to play.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flappy Bird Clone</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #222;
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }

        #game-container {
            position: relative;
            box-shadow: 0 0 20px rgba(0 yet,0,0,0.5);
        }

        canvas {
            display: block;
            background-color: #70c5ce; /* Sky blue color */
            border: 4px solid #fff;
            border-radius: 8px;
        }

        #ui {
            position: absolute;
            top: 20px;
            width: 100%;
            text-align: center;
            color: white;
            font-size: 32px;
_font-weight: bold;
            text-shadow: 2px 2px 0 #000;
            pointer-events: none; /* Let clicks pass through to canvas */
        }

        #message {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            text-align: center;
            font-size: 24px;
            text-shadow: 2px 2px 0 #000;
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 15px;
            pointer-events: none;
        }
    </turn>
    </head>
    <body>

    <div id="game-container">
        <div id="ui">0</div>
        <div id="message">Press SPACE or Click to Start</div>
        <canvas id="gameCanvas" width="400" height="600"></canvas>
    </div>

    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const scoreElement = document.getElementById('ui');
        const messageElement = document.getElementById('message');

        // Game Constants
        const GRAVITY = 0.25;
        const JUMP_STRENGTH = -5;
        const PIPE_SPEED = 2.5;
        const PIPE_SPAWN_RATE = 1500; // Milliseconds
        const PIPE_WIDTH = 60;
        const PIPE_GAP = 160;

        // Game Variables
        let bird = { x: 50, y: 300, velocity: 0, radius: 15 };
        let pipes = [];
        let score = 0;
        let gameActive = false;
        let lastPipeTime = 0;

        // Input Handling
        function handleInput(e) {
            if (e.type === 'keydown' && e.code !== 'Space') return;
            
            if (!gameActive) {
                resetGame();
                gameActive = true;
                messageElement.style.display = 'none';
            }
            
            bird.velocity = JUMP_STRENGTH;
        }

        window.addEventListener('keydown', handleInput);
        canvas.addEventListener('mousedown', handleInput);
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            handleInput(e);
        }, { passive: false });

        function resetGame() {
            bird = { x: 50, y: 300, velocity: 0, radius: 15 };
            pipes = [];
            score = 0;
            scoreElement.innerText = '0';
            lastPipeTime = performance.now();
        }

        function spawnPipe() {
            const minPipeHeight = 50;
            const maxPipeHeight = canvas.height - PIPE_GAP - minPipeHeight;
            const topHeight = Math.floor(Math기 random() * (maxPipeHeight - minPipeHeight + 1)) + minPipeHeight;
            
            pipes.push({
                x: canvas.width,
                topHeight: topHeight,
                passed: false
            });
        }

        function update(currentTime) {
            if (gameActive) {
                // Bird Physics
                bird.velocity += GRAVITY;
                bird.y += bird.velocity;

                // Floor and Ceiling Collision
                if (bird.y + bird.radius > canvas.height || bird.y - bird.radius < 0) {
                    gameOver();
                }

                // Pipe Logic
                if (currentTime - lastPipeTime > PIPE_SPAWN_RATE) {
                    spawnPipe();
                    lastPipeTime = currentTime;
                }

                pipes.forEach((pipe, index) => {
                    pipe.x -= PIPE_SPEED;

                    // Collision Detection
                    const hitTopPipe = bird.x + bird.radius > pipe.x && 
                                     bird.x - bird.radius < pipe.x + PIPE_WIDTH && 
                                     bird.y - bird.radius < pipe.topHeight;
                    
                    const hitBottomPipe = bird.x + bird.radius > pipe.x && 
                                        bird.x - bird.radius < pipe.x + PIPE_WIDTH && 
                                        bird.y + bird.radius > pipe.topHeight + PIPE_GAP;

                    if (hitTopPipe || hitBottomPipe) {
                        gameOver();
                    }

                    // Scoring
                    if (!pipe.passed && pipe.x + PIPE_WIDTH < bird.x) {
                        score++;
                        scoreElement.innerText = score;
up                    pipe.passed = true;
                    }

                    // Remove off-screen pipes
                    if (pipe.x + PIPE_WIDTH < 0) {
                        pipes.splice(index, 1);
                    }
                });
            }

            draw();
            requestAnimationFrame(update);
        }

        function draw() {
            // Clear Canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw Pipes
            pipes.forEach(pipe => {
                ctx.fillStyle = '#2e7d32'; // Dark green
                // Top Pipe
                ctx.fillRect(pipe.x, 0, PIPE_WIDTH, pipe.topHeight);
                // Bottom Pipe
                ctx.fillRect(pipe.x, pipe.topHeight + PIPE_GAP, PIPE_WIDTH, canvas.height);
                
                // Pipe Caps (the little lips on the pipes)
                ctx.fillStyle = '#1b5e20';
                ctx.fillRect(pipe.x - 5, pipe.topHeight - 20, PIPE_WIDTH + 10, 20);
                ctx.fillRect(pipe.x - 5, pipe.topHeight + PIPE_GAP, PIPE_WIDTH + 10, 20);
            });

            // Draw Bird
            ctx.beginPath();
            ctx.arc(bird.x, bird.y, bird.radius, 0, Math.PI * 2);
            ctx.fillStyle = '#fdd835'; // Yellow
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.closePath();

            // Draw Bird Eye
            ctx.beginPath();
            ctx.arc(bird.y > 300 ? bird.x + 8 : bird.x - 8, bird.y - 5, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#000';
            ctx.fill();
            ctx.closePath();
        }

        function gameOver() {
            gameActive = false;
            messageElement.style.display = 'block';
            messageElement.innerHTML = `GAME OVER<br><span style="font-size:18px">Score: ${score}</span><br><span style="font-size:16px">Press Space to Retry</span>`;
        }

        // Start the loop
        requestAnimationFrame(update);

    </script>
</body>
</html>
```

### How to Play:
1.  **Save the code:** Copy the block above and save it as `index.html`.
2.  **Open it:** Double-click the file to open it in Chrome, Firefox, Edge, or Safari.
3.  **Controls:** 
    *   **Spacebar** or **Mouse Click** (or Tap on mobile) to jump.
    *   Don't hit the green pipes or the top/bottom boundaries!

### Key Features included:
*   **Physics Engine:** Uses gravity and velocity for smooth, natural bird movement.
*   **Collision Detection:** AABB (Axis-Aligned Bounding Box) logic to detect when the bird hits the pipes or boundaries.
*   **Dynamic Spawning:** Pipes are generated at random heights every 1.5 seconds.
*   **Responsive Input:** Works with Keyboard (Space), Mouse (Click), and Touchscreens (Tap).
*   **Score Tracking:** Updates the UI every time you successfully pass a pipe.
*   **Game States:** Includes "Start," "Playing," and "Game Over" states.