pipeline {
    agent any


    stages {

        stage('GPU Sanity Test') {
            steps {
                echo 'Installing required dependencies for cuda_test'
                sh 'python3 -m pip install --upgrade pip'
                sh 'python3 -m pip install numba numpy'
                echo 'Running CUDA sanity check...'
                sh 'python3 cuda_test.py'
            }
        }


        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                sh 'docker build -t gpu-service .'
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                sh 'docker stop gpu-service || true'
                sh 'docker rm gpu-service || true'
                sh 'docker run --gpus all -d --name gpu-service -p 8001:8001 gpu-service'
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
        }
        failure {
            echo "💥 Deployment failed. Check logs for errors."
        }
        always {
            echo "🧾 Pipeline finished."
        }
    }
}
