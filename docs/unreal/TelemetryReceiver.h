#pragma once

// Minimal Unreal C++ example for receiving the existing AIDA telemetry feed over WebSocket.
// Drop these into a UE C++ project (with WebSockets plugin enabled) and attach the actor to your level.

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TelemetryReceiver.generated.h"

USTRUCT(BlueprintType)
struct FTelemetryState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) FVector Position = FVector::ZeroVector;           // meters, X fwd, Y right, Z up
    UPROPERTY(BlueprintReadOnly) FQuat Orientation = FQuat::Identity;             // quaternion w,x,y,z
    UPROPERTY(BlueprintReadOnly) FVector Velocity = FVector::ZeroVector;          // m/s
    UPROPERTY(BlueprintReadOnly) FVector Rates = FVector::ZeroVector;             // rad/s (p,q,r)
    UPROPERTY(BlueprintReadOnly) FVector Surfaces = FVector::ZeroVector;          // rad (elev, ail, rud)
    UPROPERTY(BlueprintReadOnly) float Throttle = 0.0f;                           // 0..1
    UPROPERTY(BlueprintReadOnly) float Soc = 1.0f;                                // 0..1
    UPROPERTY(BlueprintReadOnly) float Voltage = 0.0f;                            // V
    UPROPERTY(BlueprintReadOnly) float LoadFactor = 1.0f;                         // g
    UPROPERTY(BlueprintReadOnly) int32 Heartbeat = 0;
    UPROPERTY(BlueprintReadOnly) float SimTime = 0.0f;                            // seconds
    UPROPERTY(BlueprintReadOnly) FString Mode;
};

UCLASS()
class ATelemetryReceiver : public AActor
{
    GENERATED_BODY()

public:
    ATelemetryReceiver();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;

    // Host/port for the telemetry server (ws://127.0.0.1:8765 by default).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Telemetry")
    FString Host = TEXT("127.0.0.1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Telemetry")
    int32 Port = 8765;

    // Last received state (thread-safe copy on tick).
    UPROPERTY(BlueprintReadOnly, Category = "Telemetry")
    FTelemetryState LatestState;

private:
    void Connect();
    void OnMessage(const FString& Message);
    void OnConnected();
    void OnClosed(int32 StatusCode, const FString& Reason, bool bWasClean);
    void OnError(const FString& Error);

    FCriticalSection StateMutex;
    FTelemetryState PendingState;
    TSharedPtr<class IWebSocket> Socket;
    bool bHasPending = false;
};
