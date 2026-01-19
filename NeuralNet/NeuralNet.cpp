#include "framework.h"
#include "NeuralNet.h"
#include "NeuNetCode.h"
#include <vector>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>

// --- GLOBALS ---
Network* myNet = nullptr;
int selectedLayerIndex = 0;
int selectedNeuronIndex = 0;
bool isTraining = false;

struct DataPoint {
    float x, y;
    int label; // -1 = Red, 1 = Blue
};
std::vector<DataPoint> trainingData;
std::vector<DataPoint> testData;

// Screen Settings
const int GRAPH_CENTER_X = 500;
const int GRAPH_CENTER_Y = 300;
const int SCALE = 80;

// Colors
const COLORREF COL_PINK = RGB(255, 192, 203);
const COLORREF COL_BABYBLUE = RGB(173, 216, 230);
const COLORREF COL_AXIS = RGB(50, 50, 50);

// --- HELPER TO GENERATE SPIRAL POINT ---
DataPoint GeneratePoint(float rotationOffset) {
    DataPoint p;

    // Randomly choose a class: -1 (Red) or 1 (Blue)
    int type = rand() % 2;
    p.label = (type == 0) ? -1 : 1;

    // Random Radius
    float r = (float)rand() / RAND_MAX * 2.0f;

    // Spiral Math + Random Rotation
    // "type" (0 or 1) determines which arm of the spiral
    float angle = 3.5f * r + (type * 3.14159f) + rotationOffset;

    // Add noise
    float noise = ((float)rand() / RAND_MAX * 0.2f) - 0.1f;
    angle += noise;

    p.x = r * cos(angle);
    p.y = r * sin(angle);

    return p;
}

void InitScene(bool resetNetwork = true) {
    if (resetNetwork) {
        if (myNet) delete myNet;

        // Deep Structure: 20 -> 20 -> Output
        std::vector<int> structure = { 20, 20 };
        myNet = new Network(structure, 1, 2);

        selectedLayerIndex = 0;
        selectedNeuronIndex = 0;
        testData.clear();
    }

    trainingData.clear();

    // Randomize the spiral rotation slightly so it looks different every reset
    float randomRot = ((float)rand() / RAND_MAX) * 6.28f;

    for (int i = 0; i < 500; i++) {
        trainingData.push_back(GeneratePoint(randomRot));
    }
}

// RESTORED: The function to add test points
void AddTestPoints() {
    int added = 0;
    while (added < 20) {
        // Use 0.0 rotation for test points to match current spiral
        // (Or we could store the current rotation, but random is fine for testing robustness)
        DataPoint p = GeneratePoint(0.0f);
        testData.push_back(p);
        added++;
    }
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
    srand((unsigned int)time(NULL));
    InitScene(true);

    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_NEURALNET, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);
    if (!InitInstance(hInstance, nCmdShow)) return FALSE;
    // FIX: Timer attached to window is handled in InitInstance
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
    case WM_TIMER:
        if (isTraining && myNet) {
            // Train Loop
            for (int k = 0; k < 50; k++) { // 50 steps per frame
                int idx = rand() % trainingData.size();
                DataPoint& p = trainingData[idx];
                std::vector<float> inputs = { p.x, p.y };
                std::vector<float> targets = { (float)p.label };
                myNet->backPropagate(inputs, targets, 0.02f);
            }
            InvalidateRect(hWnd, NULL, FALSE);
        }
        break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);

        // 1. BACKGROUND (Step 8 for speed)
        for (int sy = 0; sy < 600; sy += 8) {
            for (int sx = 200; sx < 800; sx += 8) {
                float mathX = (sx - GRAPH_CENTER_X) / (float)SCALE;
                float mathY = (GRAPH_CENTER_Y - sy) / (float)SCALE;
                std::vector<float> output = myNet->feedForward({ mathX, mathY });

                // Threshold 0.0 for Tanh
                COLORREF color = (output[0] > 0.0f) ? COL_BABYBLUE : COL_PINK;

                HBRUSH hBrush = CreateSolidBrush(color);
                RECT rect = { sx, sy, sx + 8, sy + 8 };
                FillRect(hdc, &rect, hBrush);
                DeleteObject(hBrush);
            }
        }

        // 2. AXES
        HPEN hPenAxis = CreatePen(PS_SOLID, 2, COL_AXIS); SelectObject(hdc, hPenAxis);
        MoveToEx(hdc, 200, GRAPH_CENTER_Y, NULL); LineTo(hdc, 800, GRAPH_CENTER_Y);
        MoveToEx(hdc, GRAPH_CENTER_X, 0, NULL); LineTo(hdc, GRAPH_CENTER_X, 600);
        DeleteObject(hPenAxis);

        // 3. TRAINING DATA
        for (const auto& p : trainingData) {
            int screenX = GRAPH_CENTER_X + (int)(p.x * SCALE);
            int screenY = GRAPH_CENTER_Y - (int)(p.y * SCALE);
            // Label 1 = Blue, Label -1 = Red
            HBRUSH hBrush = (p.label == 1) ? CreateSolidBrush(RGB(0, 0, 180)) : CreateSolidBrush(RGB(180, 0, 0));
            SelectObject(hdc, hBrush); Ellipse(hdc, screenX - 4, screenY - 4, screenX + 4, screenY + 4); DeleteObject(hBrush);
        }

        // 4. TEST DATA
        HPEN hPenWhite = CreatePen(PS_SOLID, 2, RGB(255, 255, 255)); SelectObject(hdc, hPenWhite);
        for (const auto& p : testData) {
            int screenX = GRAPH_CENTER_X + (int)(p.x * SCALE);
            int screenY = GRAPH_CENTER_Y - (int)(p.y * SCALE);
            HBRUSH hBrush = (p.label == 1) ? CreateSolidBrush(RGB(0, 100, 255)) : CreateSolidBrush(RGB(255, 50, 50));
            SelectObject(hdc, hBrush); Ellipse(hdc, screenX - 7, screenY - 7, screenX + 7, screenY + 7); DeleteObject(hBrush);
        }
        DeleteObject(hPenWhite);

        // 5. GUI
        RECT textRect = { 0, 0, 200, 600 };
        FillRect(hdc, &textRect, (HBRUSH)GetStockObject(WHITE_BRUSH));
        int yPos = 10;

        TextOut(hdc, 10, yPos, L"--- SPIRAL NET ---", 18); yPos += 25;
        TextOut(hdc, 10, yPos, L"[SPACE] Reset / Rotate", 22); yPos += 20;
        TextOut(hdc, 10, yPos, L"[T]rain (Toggle)", 16); yPos += 20;
        TextOut(hdc, 10, yPos, L"[R] Add Test Points", 19); yPos += 30;

        // Deep Net Controls
        TextOut(hdc, 10, yPos, L"[Arrows] Select Layer", 21); yPos += 30;

        if (myNet) {
            // Stats
            std::wstring s1 = L"Training: " + std::to_wstring(trainingData.size());
            TextOut(hdc, 10, yPos, s1.c_str(), s1.length()); yPos += 20;
            std::wstring s2 = L"Test Pts: " + std::to_wstring(testData.size());
            TextOut(hdc, 10, yPos, s2.c_str(), s2.length()); yPos += 30;

            // Layer Header
            std::wstring lHeader = L"LAYER " + std::to_wstring(selectedLayerIndex);
            if (selectedLayerIndex == 0) lHeader += L" (In)";
            else if (selectedLayerIndex == myNet->layers.size() - 1) lHeader += L" (Out)";
            TextOut(hdc, 10, yPos, lHeader.c_str(), lHeader.length()); yPos += 20;

            // Neurons
            auto& activeLayer = myNet->layers[selectedLayerIndex];
            for (size_t i = 0; i < activeLayer.neurons.size(); i++) {
                // Only show first 12 neurons to fit on screen
                if (i > 12) { TextOut(hdc, 10, yPos, L"...", 3); break; }

                std::wstringstream ss;
                ss.precision(2);
                if (i == selectedNeuronIndex) { ss << L">"; SetTextColor(hdc, RGB(200, 0, 0)); }
                else { ss << L" "; SetTextColor(hdc, RGB(0, 0, 0)); }

                ss << L"N" << i << L" b:" << activeLayer.neurons[i].bias;
                if (activeLayer.neurons[i].weights.size() > 0)
                    ss << L" w:" << activeLayer.neurons[i].weights[0];

                std::wstring s = ss.str();
                TextOut(hdc, 10, yPos, s.c_str(), s.length()); yPos += 20;
            }
            SetTextColor(hdc, RGB(0, 0, 0));
        }

        EndPaint(hWnd, &ps);
    }
    break;

    case WM_KEYDOWN:
    {
        if (!myNet) break;
        int maxLayers = myNet->layers.size();
        int maxNeurons = myNet->layers[selectedLayerIndex].neurons.size();
        Node* activeNode = &myNet->layers[selectedLayerIndex].neurons[selectedNeuronIndex];
        float step = 0.1f;

        switch (wParam) {
        case VK_SPACE: InitScene(true); InvalidateRect(hWnd, NULL, FALSE); break;
        case 'T': isTraining = !isTraining; break;

            // FIX: The 'R' key logic is back!
        case 'R': AddTestPoints(); InvalidateRect(hWnd, NULL, FALSE); break;

            // Navigation
        case VK_RIGHT:
            selectedLayerIndex++; if (selectedLayerIndex >= maxLayers) selectedLayerIndex = 0;
            selectedNeuronIndex = 0; break;
        case VK_LEFT:
            selectedLayerIndex--; if (selectedLayerIndex < 0) selectedLayerIndex = maxLayers - 1;
            selectedNeuronIndex = 0; break;
        case VK_DOWN:
            selectedNeuronIndex++; if (selectedNeuronIndex >= maxNeurons) selectedNeuronIndex = 0; break;
        case VK_UP:
            selectedNeuronIndex--; if (selectedNeuronIndex < 0) selectedNeuronIndex = maxNeurons - 1; break;

            // Tweaking
        case 'Q': if (!activeNode->weights.empty()) activeNode->weights[0] += step; break;
        case 'A': if (!activeNode->weights.empty()) activeNode->weights[0] -= step; break;
        case 'E': activeNode->bias += step; break;
        case 'D': activeNode->bias -= step; break;
        }
        InvalidateRect(hWnd, NULL, FALSE);
    }
    break;

    case WM_DESTROY: PostQuitMessage(0); break;
    default: return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// (InitInstance/MyRegisterClass unchanged)
ATOM MyRegisterClass(HINSTANCE hInstance) { WNDCLASSEXW wcex; wcex.cbSize = sizeof(WNDCLASSEX); wcex.style = CS_HREDRAW | CS_VREDRAW; wcex.lpfnWndProc = WndProc; wcex.cbClsExtra = 0; wcex.cbWndExtra = 0; wcex.hInstance = hInstance; wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_NEURALNET)); wcex.hCursor = LoadCursor(nullptr, IDC_ARROW); wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_NEURALNET); wcex.lpszClassName = szWindowClass; wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL)); return RegisterClassExW(&wcex); }
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) { hInst = hInstance; HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, 850, 650, nullptr, nullptr, hInstance, nullptr); if (!hWnd) return FALSE; SetTimer(hWnd, 1, 10, nullptr); ShowWindow(hWnd, nCmdShow); UpdateWindow(hWnd); return TRUE; }