@echo off
echo Creating Connect4 AI Coursework Project Structure...

:: Create main directory (if not already in it)
mkdir connect4_ai_coursework

:: Create subdirectories
mkdir connect4_ai_coursework\agents
mkdir connect4_ai_coursework\ml
mkdir connect4_ai_coursework\evaluation
mkdir connect4_ai_coursework\utils

:: Create empty files
echo # Entry point to run the game > connect4_ai_coursework\main.py
echo # Game logic (board, moves, win/draw check) > connect4_ai_coursework\connect4.py

:: Agent files
echo # Random agent implementation > connect4_ai_coursework\agents\random_agent.py
echo # Rule-based smart agent > connect4_ai_coursework\agents\smart_agent.py
echo # Minimax (with/without alpha-beta) > connect4_ai_coursework\agents\minimax_agent.py
echo # Machine Learning agent > connect4_ai_coursework\agents\ml_agent.py

:: ML files
echo # Load & clean UCI dataset > connect4_ai_coursework\ml\preprocess_data.py
echo # Train the ML model > connect4_ai_coursework\ml\train_model.py
echo # Placeholder for saved model > connect4_ai_coursework\ml\model.pkl

:: Evaluation files
echo # Runs batches of games > connect4_ai_coursework\evaluation\simulate_games.py
echo # Evaluation metrics and visualizations > connect4_ai_coursework\evaluation\metrics.py

:: Utils files
echo # Any common utility functions > connect4_ai_coursework\utils\helpers.py

:: Documentation
echo # Connect4 AI Coursework > connect4_ai_coursework\README.md
echo # Project dependencies > connect4_ai_coursework\requirements.txt

echo.
echo Project structure created successfully!
echo.
echo To navigate to your project:
echo cd connect4_ai_coursework
pause