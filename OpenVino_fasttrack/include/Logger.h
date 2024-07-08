#pragma once

#define LOG_PRINT(printLev, format, ...) LogPrinter::LogPrint(printLev, __FILE__, __LINE__, format, __VA_ARGS__);

#define LOG_PRINT_INFO(format, ...) LogPrinter::LogPrint(PRINT_LEV_INFO, __FILE__, __LINE__, format, __VA_ARGS__);
#define LOG_PRINT_EVENT(format, ...) LogPrinter::LogPrint(PRINT_LEV_EVENT, __FILE__, __LINE__, format, __VA_ARGS__);
#define LOG_PRINT_ERROR(format, ...) LogPrinter::LogPrint(PRINT_LEV_ERROR, __FILE__, __LINE__, format, __VA_ARGS__);
#define LOG_PRINT_DEBUG(format, ...) LogPrinter::LogPrint(PRINT_LEV_DEBUG, __FILE__, __LINE__, format, __VA_ARGS__);

enum PRINT_LEV
{
    PRINT_LEV_DEBUG,
    PRINT_LEV_INFO,
    PRINT_LEV_EVENT,
    PRINT_LEV_ERROR,
    PRINT_LEV_UNKNOWN,
    PRINT_LEV_NUM
};

class LogPrinter
{
private:
    explicit LogPrinter();
    ~LogPrinter();

public:
    static void LogPrint(PRINT_LEV printLev, const char* filename, int lineNo, const char* format, ...);

private:
    class LogPrinterDef;
    static LogPrinterDef* _PrinterDef;
};