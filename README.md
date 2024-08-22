# Screenshot Timeline

A work in progress privacy focused alternative to Microsoft Copilot and similar spyware.

Screenshot Timeline is an automated screenshot capture and analysis tool that uses OCR to extract text from images and organize them with tags for easy searching and filtering.

Thanks to Screenshot Timeline, you can now easily search through your screenshots and find specific ones based on the text they contain.

## Features

- Automated screenshot capture at customizable intervals
- OCR processing using Tesseract
- Tag generation based on extracted text
- Search functionality for finding specific screenshots
- Real-time updates of screenshot processing status
- Ability to edit tags for each screenshot
- Batch processing of unanalyzed screenshots
- Configuration options for OCR processing and screenshot management

## Technology Stack

- Backend: Python with Flask
- Frontend: HTML, CSS, JavaScript
- OCR: Tesseract
- Task Queue: Celery with Redis
- Database: SQLite
- Image Processing: OpenCV

## Why Tesseract?

We chose Tesseract for OCR processing due to its:

1. Open-source nature and active community support
2. High accuracy in text recognition across various languages
3. Easy integration with Python through the pytesseract library
4. Flexibility in configuration for different use cases
5. Continuous improvements and updates

## Installation and Setup

1. Update and upgrade your system:
```
sudo apt update && sudo apt upgrade -y
```

2. Install required system dependencies:
```
sudo apt install -y python3 python3-pip redis-server spectacle tesseract-ocr libtesseract-dev
```

3. Clone the repository:
git clone https://github.com/tcsenpai/screenshot-timeline.git
cd screenshot-timeline


4. Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate
```

5. Install Python dependencies:
```
pip install -r requirements.txt
```

6. Start the app:
```
./start_services.sh
```

7. Open your browser and go to http://localhost:5000


## Configuration

The application can be configured through the web interface. Options include:

- Setting screenshot capture intervals
- Triggering OCR for all unprocessed images
- Deleting all screenshots and resetting the database

## Contributing


We welcome contributions to the Screenshot Timeline project! If you'd like to contribute, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name for your feature or bug fix.
3. Make your changes, ensuring you follow the project's coding style and conventions.
4. Add or update tests as necessary to cover your changes.
5. Ensure all tests pass by running the test suite.
6. Commit your changes with a clear and descriptive commit message.
7. Push your branch to your fork on GitHub.
8. Open a pull request against the main repository's `main` branch.
9. Provide a clear description of your changes in the pull request.

Please note that by contributing to this project, you agree to license your contributions under the project's MIT License. For major changes or new features, please open an issue first to discuss the proposed changes. We strive to maintain a welcoming and inclusive community, so please adhere to our code of conduct in all interactions related to the project.

## License

This project is licensed under the MIT License.

## Credits

- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Flask: https://flask.palletsprojects.com/
- Celery: https://docs.celeryproject.org/
- OpenCV: https://opencv.org/
- Material Design Components: https://material.io/develop/web

(Add any other libraries or resources used in the project)