#include "framework.h"
#include "NeuralNet.h"
#include "NeuNetCode.h"
#include <vector>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>

// --- SETTINGS & GLOBALS ---
Network* myNet = nullptr;
int hiddenNeuronCount = 3; // Start with 3 for a triangle shape
int selectedNeuronIndex = 0; // Which neuron are we editing?
bool isTraining = false;

struct DataPoint {
    float x, y;
    int label; // 0 = Red, 1 = Blue
};
std::vector<DataPoint> trainingData;

// Screen Settings
const int GRAPH_CENTER_X = 500;
const int GRAPH_CENTER_Y = 300;
const int SCALE = 80;

// Colors
const COLORREF COL_PINK = RGB(255, 192, 203);
const COLORREF COL_BABYBLUE = RGB(173, 216, 230);
const COLORREF COL_AXIS = RGB(50, 50, 50);
const COLORREF COL_HIGHLIGHT = RGB(255, 215, 0); // Gold for selected text

// --- WINDOWS BOILERPLATE ---
#define MAX_LOADSTRING 100
HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];
ATOM MyRegisterClass(HINSTANCE hInstance);
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// --- INIT DATA ---
void InitScene(bool resetNetwork = true) {

    // 1. Only rebuild network if requested (e.g. SPACE bar)
    if (resetNetwork) {
        if (myNet) delete myNet;
        std::vector<int> structure = { hiddenNeuronCount };
        myNet = new Network(structure, 1, 2);
        selectedNeuronIndex = 0; // Reset selection
    }

    // 2. Generate Data (100 Points)
    trainingData.clear();
    for (int i = 0; i < 100; i++) {
        DataPoint p;
        p.x = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        p.y = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;

        float dist = sqrt(p.x * p.x + p.y * p.y);

        // Bullseye Logic
        if (dist < 0.8f) p.label = 0;      // Inner Red
        else if (dist > 1.2f) p.label = 1; // Outer Blue
        else continue; // Skip the gap

        trainingData.push_back(p);
    }
}

int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE hPrev, LPWSTR lpCmdLine, int nCmdShow) {
    srand((unsigned int)time(NULL));
    InitScene(true); // First launch = full reset

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
    case WM_TIMER:
        if (isTraining && myNet) {
            // Train on 50 random points per frame (Speed up learning)
            for (int k = 0; k < 50; k++) {
                // Pick a random data point
                int idx = rand() % trainingData.size();
                DataPoint& p = trainingData[idx];

                std::vector<float> inputs = { p.x, p.y };
                std::vector<float> targets = { (float)p.label }; // 0.0 or 1.0

                // BACKPROPAGATE!
                // Learning Rate 0.05 is usually safe for this size
                myNet->backPropagate(inputs, targets, 0.25f);
            }
            // Trigger a redraw to see the progress
            InvalidateRect(hWnd, NULL, FALSE);
        }
        break;
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);

        // --- 1. DRAW BACKGROUND ---
        for (int sy = 0; sy < 600; sy += 8) {
            for (int sx = 200; sx < 800; sx += 8) {
                float mathX = (sx - GRAPH_CENTER_X) / (float)SCALE;
                float mathY = (GRAPH_CENTER_Y - sy) / (float)SCALE;

                std::vector<float> input = { mathX, mathY };
                std::vector<float> output = myNet->feedForward(input);

                COLORREF color = (output[0] > 0.5f) ? COL_BABYBLUE : COL_PINK;
                HBRUSH hBrush = CreateSolidBrush(color);
                RECT rect = { sx, sy, sx + 8, sy + 8 };
                FillRect(hdc, &rect, hBrush);
                DeleteObject(hBrush);
            }
        }

        // --- 2. DRAW AXES ---
        HPEN hPenAxis = CreatePen(PS_SOLID, 2, COL_AXIS);
        SelectObject(hdc, hPenAxis);
        MoveToEx(hdc, 200, GRAPH_CENTER_Y, NULL); LineTo(hdc, 800, GRAPH_CENTER_Y);
        MoveToEx(hdc, GRAPH_CENTER_X, 0, NULL); LineTo(hdc, GRAPH_CENTER_X, 600);
        DeleteObject(hPenAxis);

        // --- 3. DRAW DATA POINTS ---
        for (const auto& p : trainingData) {
            int screenX = GRAPH_CENTER_X + (int)(p.x * SCALE);
            int screenY = GRAPH_CENTER_Y - (int)(p.y * SCALE);
            HBRUSH hBrush = (p.label == 1) ? CreateSolidBrush(RGB(0, 0, 255)) : CreateSolidBrush(RGB(255, 0, 0));
            SelectObject(hdc, hBrush);
            Ellipse(hdc, screenX - 6, screenY - 6, screenX + 6, screenY + 6);
            DeleteObject(hBrush);
        }

        // --- 4. GUI & MANUAL CONTROLS ---
        RECT textRect = { 0, 0, 200, 600 };
        FillRect(hdc, &textRect, (HBRUSH)GetStockObject(WHITE_BRUSH));

        int yPos = 10;
        TextOut(hdc, 10, yPos, L"--- CONTROLS ---", 16); yPos += 20;
        TextOut(hdc, 10, yPos, L"SPACE: New Network", 18); yPos += 20;
        TextOut(hdc, 10, yPos, L"TAB: Select Next Neuron", 23); yPos += 20;
        TextOut(hdc, 10, yPos, L"Q/A: Tweak Weight X", 19); yPos += 20;
        TextOut(hdc, 10, yPos, L"W/S: Tweak Weight Y", 19); yPos += 20;
        TextOut(hdc, 10, yPos, L"E/D: Tweak Bias", 15); yPos += 30;

        // FIX: Define string first, pass length correctly
        std::wstring tMsg = L"Press 'T' to Auto-Train";
        TextOut(hdc, 10, yPos, tMsg.c_str(), tMsg.length());
        yPos += 35; // Increment line AFTER

        std::wstring status = L"Hidden Neurons: " + std::to_wstring(hiddenNeuronCount);
        TextOut(hdc, 10, yPos, status.c_str(), status.length());
        yPos += 40;

        // Display Neurons
        auto& hiddenLayer = myNet->layers[0];

        for (size_t i = 0; i < hiddenLayer.neurons.size(); i++) {
            std::wstringstream ss;
            ss.precision(2);
            // Highlight selected neuron
            if (i == selectedNeuronIndex) ss << L"> N" << i;
            else ss << L"  N" << i;

            ss << L" w:[" << hiddenLayer.neurons[i].weights[0] << L"," << hiddenLayer.neurons[i].weights[1] << L"]";
            ss << L" b:" << hiddenLayer.neurons[i].bias;

            std::wstring s = ss.str();

            // Set text color if selected
            if (i == selectedNeuronIndex) SetTextColor(hdc, RGB(200, 0, 0)); // Red text for active
            else SetTextColor(hdc, RGB(0, 0, 0));

            TextOut(hdc, 10, yPos, s.c_str(), s.length());
            yPos += 20;
        }
        SetTextColor(hdc, RGB(0, 0, 0)); // Reset

        EndPaint(hWnd, &ps);
    }
    break;

    case WM_KEYDOWN:
    {
        // Pointer to the specific neuron we are editing
        Node* activeNode = &myNet->layers[0].neurons[selectedNeuronIndex];
        float step = 0.1f;

        switch (wParam) {
            // --- GLOBAL CONTROLS ---
        case VK_SPACE: InitScene(true); break; // Reset Net
        case VK_TAB:   // Cycle Selection
            selectedNeuronIndex++;
            if (selectedNeuronIndex >= hiddenNeuronCount) selectedNeuronIndex = 0;
            break;
        case VK_UP:
            hiddenNeuronCount++; if (hiddenNeuronCount > 10) hiddenNeuronCount = 10;
            InitScene(true);
            break;
        case VK_DOWN:
            hiddenNeuronCount--; if (hiddenNeuronCount < 1) hiddenNeuronCount = 1;
            InitScene(true);
            break;

            // --- FINE TUNING CONTROLS ---
        case 'Q': activeNode->weights[0] += step; break; // X Weight UP
        case 'A': activeNode->weights[0] -= step; break; // X Weight DOWN

        case 'W': activeNode->weights[1] += step; break; // Y Weight UP
        case 'S': activeNode->weights[1] -= step; break; // Y Weight DOWN

        case 'E': activeNode->bias += step; break; // Bias UP
        case 'D': activeNode->bias -= step; break; // Bias DOWN

        case 'T':
            isTraining = !isTraining; // Toggle ON/OFF
            break;
        }

        InvalidateRect(hWnd, NULL, FALSE); // Redraw
    }
    break;

    case WM_DESTROY: PostQuitMessage(0); break;
    default: return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// (Keep InitInstance, MyRegisterClass unchanged)
ATOM MyRegisterClass(HINSTANCE hInstance) { WNDCLASSEXW wcex; wcex.cbSize = sizeof(WNDCLASSEX); wcex.style = CS_HREDRAW | CS_VREDRAW; wcex.lpfnWndProc = WndProc; wcex.cbClsExtra = 0; wcex.cbWndExtra = 0; wcex.hInstance = hInstance; wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_NEURALNET)); wcex.hCursor = LoadCursor(nullptr, IDC_ARROW); wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_NEURALNET); wcex.lpszClassName = szWindowClass; wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL)); return RegisterClassExW(&wcex); }
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) {
    hInst = hInstance;
    HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, 850, 650, nullptr, nullptr, hInstance, nullptr);

    if (!hWnd) return FALSE;

    SetTimer(hWnd, 1, 10, nullptr);

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);
    return TRUE;
}