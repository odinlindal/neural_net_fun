
#include "framework.h"
#include "NeuralNet.h"
#include "NeuNetCode.h"
#include "MnistLoader.h"
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip>

// --- HELPER: CENTER THE IMAGE ---
// This moves the drawing so its center of mass aligns with the grid center (14,14)
void CenterGrid(float* inputGrid, float* outputGrid) {
    // 1. Calculate Center of Mass
    float sumX = 0.0f, sumY = 0.0f, totalWeight = 0.0f;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float val = inputGrid[y * 28 + x];
            if (val > 0.0f) {
                sumX += x * val;
                sumY += y * val;
                totalWeight += val;
            }
        }
    }

    // If grid is empty, just copy zeros
    if (totalWeight == 0.0f) {
        for (int i = 0; i < 784; i++) outputGrid[i] = 0.0f;
        return;
    }

    // Current Center
    float centerX = sumX / totalWeight;
    float centerY = sumY / totalWeight;

    // Target Center is (14, 14)
    float shiftX = 14.0f - centerX;
    float shiftY = 14.0f - centerY;

    // 2. Shift Pixels
    for (int i = 0; i < 784; i++) outputGrid[i] = 0.0f; // Clear output

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float val = inputGrid[y * 28 + x];
            if (val > 0.0f) {
                // Apply shift
                int newX = (int)(x + shiftX);
                int newY = (int)(y + shiftY);

                // Check bounds
                if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28) {
                    outputGrid[newY * 28 + newX] = val;
                }
            }
        }
    }
}

// --- GLOBALS ---
Network* myNet = nullptr;
float drawingGrid[784]; // The 28x28 canvas (0.0 = Black, 1.0 = White)
bool isDrawing = false;
std::vector<float> currentOutputs;

// Screen Settings
const int GRID_OFFSET_X = 50;
const int GRID_OFFSET_Y = 50;
const int CELL_SIZE = 15; // Make pixels big enough to see

// --- HELPER: RESET GRID ---
void ClearGrid() {
    for (int i = 0; i < 784; i++) drawingGrid[i] = 0.0f;
}

// --- HELPER: PAINT ON GRID (With a little "brush" thickness) ---
void PaintGrid(int mouseX, int mouseY) {
    // Convert Screen Coords -> Grid Coords
    int gx = (mouseX - GRID_OFFSET_X) / CELL_SIZE;
    int gy = (mouseY - GRID_OFFSET_Y) / CELL_SIZE;

    if (gx >= 0 && gx < 28 && gy >= 0 && gy < 28) {
        // Center pixel = White (1.0)
        drawingGrid[gy * 28 + gx] = 1.0f;

        // Neighbors = Gray (0.5) to simulate thickness (Anti-aliasingish)
        // MNIST digits are thick, so single-pixel lines perform poorly.
        int neighbors[4][2] = { {0,1}, {0,-1}, {1,0}, {-1,0} };
        for (auto& n : neighbors) {
            int nx = gx + n[0];
            int ny = gy + n[1];
            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                // Don't overwrite if it's already brighter
                if (drawingGrid[ny * 28 + nx] < 0.5f) {
                    drawingGrid[ny * 28 + nx] = 0.5f;
                }
            }
        }
    }
}

// --- INIT ---
void InitScene() {
    if (myNet) delete myNet;

    // 1. Setup Structure
    std::vector<int> structure = { 100 };
    myNet = new Network(structure, 10, 784);

    // 2. Load the trained brain
    // Make sure 'brain.txt' is in the same folder as the .exe (or project folder)
    if (!myNet->loadNetwork("brain.txt")) {
        MessageBox(NULL, L"Could not load 'brain.txt'! Did you run the trainer?", L"Brain Missing", MB_ICONWARNING);
    }

    ClearGrid();
}

// --- WINDOWS BOILERPLATE ---
#define MAX_LOADSTRING 100
HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];
ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE hPrev, LPWSTR lpCmdLine, int nCmdShow) {
    InitScene();
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_NEURALNET, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);
    if (!InitInstance(hInstance, nCmdShow)) return FALSE;
    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_NEURALNET));
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
    return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);

        // 1. DRAW THE GRID (28x28)
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int screenX = GRID_OFFSET_X + x * CELL_SIZE;
                int screenY = GRID_OFFSET_Y + y * CELL_SIZE;

                float val = drawingGrid[y * 28 + x];
                int gray = (int)(val * 255.0f);

                // Draw pixel
                HBRUSH hBrush = CreateSolidBrush(RGB(gray, gray, gray));
                RECT rect = { screenX, screenY, screenX + CELL_SIZE, screenY + CELL_SIZE };
                FillRect(hdc, &rect, hBrush);
                DeleteObject(hBrush);

                // Grid lines (Optional, subtle)
                // SetPixel(hdc, screenX, screenY, RGB(50,50,50)); 
            }
        }

        // Border around grid
        HBRUSH frameBrush = CreateSolidBrush(RGB(255, 0, 0));
        RECT frameRect = { GRID_OFFSET_X - 2, GRID_OFFSET_Y - 2, GRID_OFFSET_X + 28 * CELL_SIZE + 2, GRID_OFFSET_Y + 28 * CELL_SIZE + 2 };
        FrameRect(hdc, &frameRect, frameBrush);
        DeleteObject(frameBrush);


        // 2. RUN PREDICTION
        if (myNet && !currentOutputs.empty()) {

            // 3. DRAW BAR CHART
            int barX = GRID_OFFSET_X + 28 * CELL_SIZE + 50;
            int barY = GRID_OFFSET_Y;

            TextOut(hdc, barX, barY - 20, L"CONFIDENCE:", 11);

            int maxIndex = 0;
            float maxVal = -1.0f;

            for (int i = 0; i < 10; i++) {
                float val = currentOutputs[i];
                if (val > maxVal) { maxVal = val; maxIndex = i; }

                // Map -1..1 (Tanh) or 0..1 (Sigmoid) to 0..200 pixels width
                // Assuming Tanh (-1 to 1), let's clamp negative to 0
                if (val < 0) val = 0;
                int width = (int)(val * 200.0f);

                // Draw Bar
                HBRUSH barBrush = (i == maxIndex && val > 0.5f) ? CreateSolidBrush(RGB(0, 200, 0)) : CreateSolidBrush(RGB(100, 100, 100));
                RECT barRect = { barX + 20, barY + i * 30, barX + 20 + width, barY + i * 30 + 20 };
                FillRect(hdc, &barRect, barBrush);
                DeleteObject(barBrush);

                // Draw Label "0", "1", ...
                std::wstring num = std::to_wstring(i);
                TextOut(hdc, barX, barY + i * 30, num.c_str(), num.length());

                // Draw Value Text
                std::wstringstream ss;
                ss << std::fixed << std::setprecision(2) << val;
                std::wstring sVal = ss.str();
                TextOut(hdc, barX + 230, barY + i * 30, sVal.c_str(), sVal.length());
            }

            // BIG RESULT
            HFONT hFontBig = CreateFont(50, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, VARIABLE_PITCH, TEXT("Arial"));
            SelectObject(hdc, hFontBig);

            std::wstring bigRes = L"Guess: " + std::to_wstring(maxIndex);
            TextOut(hdc, barX, barY + 320, bigRes.c_str(), bigRes.length());

            DeleteObject(hFontBig);
        }

        // Instructions
        TextOut(hdc, GRID_OFFSET_X, GRID_OFFSET_Y + 28 * CELL_SIZE + 20, L"Left Click: Draw   |   Right Click: Erase Pixel   |   [C]lear Canvas", 58);

        EndPaint(hWnd, &ps);
    }
    break;

    case WM_MOUSEMOVE:
        if (wParam & MK_LBUTTON) {
            PaintGrid(LOWORD(lParam), HIWORD(lParam));

            if (myNet) {
                // 1. Create a centered version
                float centeredGrid[784];
                CenterGrid(drawingGrid, centeredGrid);

                // 2. Feed the centered version to the brain
                std::vector<float> inputs;
                for (int i = 0; i < 784; i++) inputs.push_back(centeredGrid[i]);

                currentOutputs = myNet->feedForward(inputs);
            }
            InvalidateRect(hWnd, NULL, FALSE);
        }
        // Right click erase
        else if (wParam & MK_RBUTTON) {
            // Simple erase logic (set to 0)
            int gx = (LOWORD(lParam) - GRID_OFFSET_X) / CELL_SIZE;
            int gy = (HIWORD(lParam) - GRID_OFFSET_Y) / CELL_SIZE;
            if (gx >= 0 && gx < 28 && gy >= 0 && gy < 28) drawingGrid[gy * 28 + gx] = 0.0f;
            InvalidateRect(hWnd, NULL, FALSE);
        }
        break;

    case WM_KEYDOWN:
        if (wParam == 'C') {
            ClearGrid();
            currentOutputs.clear();
            InvalidateRect(hWnd, NULL, TRUE);
        }
        break;

    case WM_DESTROY: PostQuitMessage(0); break;
    default: return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// (Standard boilerplate follows)
ATOM MyRegisterClass(HINSTANCE hInstance) { WNDCLASSEXW wcex; wcex.cbSize = sizeof(WNDCLASSEX); wcex.style = CS_HREDRAW | CS_VREDRAW; wcex.lpfnWndProc = WndProc; wcex.cbClsExtra = 0; wcex.cbWndExtra = 0; wcex.hInstance = hInstance; wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_NEURALNET)); wcex.hCursor = LoadCursor(nullptr, IDC_ARROW); wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_NEURALNET); wcex.lpszClassName = szWindowClass; wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL)); return RegisterClassExW(&wcex); }
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) { hInst = hInstance; HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, 900, 600, nullptr, nullptr, hInstance, nullptr); if (!hWnd) return FALSE; ShowWindow(hWnd, nCmdShow); UpdateWindow(hWnd); return TRUE; }




/* Old console layout for training */
/*
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <algorithm> // For shuffling
#include <iomanip>   // For std::setprecision

// Include your headers
#include "NeuNetCode.h"
#include "MnistLoader.h" 

// Helper: Find the index of the highest output neuron (0-9)
int GetPrediction(const std::vector<float>& outputs) {
    int maxIndex = 0;
    float maxVal = outputs[0];
    for (int i = 1; i < outputs.size(); i++) {
        if (outputs[i] > maxVal) {
            maxVal = outputs[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

int main() {
    srand((unsigned int)time(NULL));

    std::cout << "=======================================\n";
    std::cout << "   MNIST CONSOLE TRAINER (PERSISTENT)   \n";
    std::cout << "=======================================\n";

    // 1. LOAD DATA
    std::cout << "[1/4] Loading MNIST Data... ";
    // Train Data (The Textbook)
    std::cout << "   Loading Training Set... ";
    std::vector<MnistImage> trainingData = LoadMnistData("Data\\train-images.idx3-ubyte", "Data\\train-labels.idx1-ubyte");
    std::cout << trainingData.size() << " images.\n";

    // Test Data (The Final Exam)
    std::cout << "   Loading Test Set... ";
    // Notice the filenames: t10k instead of train
    std::vector<MnistImage> testData = LoadMnistData("Data\\t10k-images.idx3-ubyte", "Data\\t10k-labels.idx1-ubyte");
    std::cout << testData.size() << " images.\n";
    std::cout << "Done!\n";

    // 2. INITIALIZE NETWORK
    std::cout << "[2/4] Setting up Network... ";
    std::vector<int> structure = { 100 };
    Network myNet(structure, 10, 784);
    std::cout << "Done.\n";

    // 3. CHECK FOR SAVED BRAIN
    std::cout << "[3/4] Checking for 'brain.txt'... ";
    if (myNet.loadNetwork("brain.txt")) {
        std::cout << "Loaded! \nDo you want to continue training? (y/n): ";
        char ans;
        std::cin >> ans;
        if (ans == 'n') {
            std::cout << "Skipping training. Jumping to test mode.\n";
            goto TEST_SECTION; // Jump to the end
        }
    }
    else {
        std::cout << "No save found. Starting fresh.\n";
    }

    // 4. TRAINING LOOP (Same as before)
    {
        int epochs = 5;
        float learningRate = 0.01f;
        int batchSize = trainingData.size();

        std::cout << "\nStarting Training...\n";

        for (int epoch = 1; epoch <= epochs; epoch++) {
            std::random_shuffle(trainingData.begin(), trainingData.end());
            int correctPredictions = 0;

            for (int i = 0; i < batchSize; i++) {
                MnistImage& img = trainingData[i];
                std::vector<float> targets(10, 0.0f);
                targets[img.label] = 1.0f;

                myNet.backPropagate(img.pixels, targets, learningRate);

                // --- RESTORED ACCURACY CHECK ---
                // 1. Ask the network what it thinks (Forward pass)
                std::vector<float> outputs = myNet.feedForward(img.pixels);

                // 2. Did it get it right?
                int prediction = GetPrediction(outputs); // Uses the helper at top of file
                if (prediction == img.label) {
                    correctPredictions++;
                }

                // 3. Print Progress every 2000 images
                if (i % 2000 == 0 && i > 0) {
                    float currentAcc = (float)correctPredictions / (i + 1) * 100.0f;
                    std::cout << "\r   Epoch " << epoch << ": "
                        << i << "/" << batchSize
                        << " | Accuracy: " << std::fixed << std::setprecision(2) << currentAcc << "%"
                        << std::flush;
                }
            }
            std::cout << " Epoch " << epoch << " Complete.\n";

            // AUTO-SAVE after every epoch (Safety)
            myNet.saveNetwork("brain.txt");
        }
    }

TEST_SECTION:
    std::cout << "\n[4/4] Network Ready.\n";

    std::cout << "\n=======================================\n";
    std::cout << "          FINAL EXAM (TEST SET)        \n";
    std::cout << "=======================================\n";

    int correct = 0;
    for (int i = 0; i < testData.size(); i++) {
        MnistImage& img = testData[i];

        // 1. Ask the network
        std::vector<float> outputs = myNet.feedForward(img.pixels);

        // 2. Check answer
        int prediction = GetPrediction(outputs);
        if (prediction == img.label) {
            correct++;
        }
    }

    float finalAcc = (float)correct / testData.size() * 100.0f;
    std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(2) << finalAcc << "%\n";

    if (finalAcc > 90.0f) std::cout << "GRADE: A (Excellent!)\n";
    else if (finalAcc > 80.0f) std::cout << "GRADE: B (Good)\n";
    else std::cout << "GRADE: F (Needs more training)\n";

    return 0;
}
*/
