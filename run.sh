#!/bin/bash

# Researchify2 Docker Setup Script

echo "🔍 Researchify2 Docker Setup"
echo "=============================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "📝 Creating .env from template..."
    
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "✅ .env created from template"
        echo "⚠️  Please edit .env with your actual API keys before running"
        echo ""
        echo "Required variables:"
        echo "  - OPENAI_API_KEY"
        echo "  - KAGGLE_USERNAME (optional)"
        echo "  - KAGGLE_KEY (optional)"
        echo ""
        read -p "Press Enter after you've updated .env with your keys..."
    else
        echo "❌ No .env.template found. Please create .env manually."
        exit 1
    fi
fi

echo "🐳 Building Docker image..."
docker build -t researchify2 .

echo "🚀 Starting application..."
echo "📱 App will be available at: http://localhost:8501"
echo ""

# Choose run method
echo "How would you like to run?"
echo "1) Foreground (see logs, Ctrl+C to stop)"
echo "2) Background (detached mode)"
echo "3) Using docker-compose"
read -p "Choose (1-3): " choice

case $choice in
    1)
        echo "🏃 Running in foreground..."
        docker run --env-file .env -p 8501:8501 researchify2
        ;;
    2)
        echo "🏃 Running in background..."
        container_id=$(docker run -d --env-file .env -p 8501:8501 researchify2)
        echo "✅ Container started: $container_id"
        echo "🛑 To stop: docker stop $container_id"
        ;;
    3)
        echo "🏃 Using docker-compose..."
        docker-compose up -d
        echo "✅ Started with docker-compose"
        echo "🛑 To stop: docker-compose down"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Researchify2 is ready!"
echo "🌐 Open: http://localhost:8501"
