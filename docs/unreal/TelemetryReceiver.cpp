#include "TelemetryReceiver.h"

#include "IWebSocket.h"
#include "WebSocketsModule.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "Misc/ScopeLock.h"

ATelemetryReceiver::ATelemetryReceiver()
{
    PrimaryActorTick.bCanEverTick = true;
}

void ATelemetryReceiver::BeginPlay()
{
    Super::BeginPlay();
    Connect();
}

void ATelemetryReceiver::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    // Copy pending state to LatestState on the game thread.
    if (bHasPending)
    {
        FScopeLock Lock(&StateMutex);
        LatestState = PendingState;
        bHasPending = false;
    }
}

void ATelemetryReceiver::Connect()
{
    if (!FModuleManager::Get().IsModuleLoaded("WebSockets"))
    {
        FModuleManager::Get().LoadModule("WebSockets");
    }

    const FString Url = FString::Printf(TEXT("ws://%s:%d"), *Host, Port);
    Socket = FWebSocketsModule::Get().CreateWebSocket(Url);

    Socket->OnConnected().AddUObject(this, &ATelemetryReceiver::OnConnected);
    Socket->OnConnectionError().AddUObject(this, &ATelemetryReceiver::OnError);
    Socket->OnClosed().AddUObject(this, &ATelemetryReceiver::OnClosed);
    Socket->OnMessage().AddUObject(this, &ATelemetryReceiver::OnMessage);

    Socket->Connect();
}

void ATelemetryReceiver::OnConnected()
{
    UE_LOG(LogTemp, Log, TEXT("[Telemetry] Connected to %s:%d"), *Host, Port);
}

void ATelemetryReceiver::OnClosed(int32 StatusCode, const FString& Reason, bool bWasClean)
{
    UE_LOG(LogTemp, Warning, TEXT("[Telemetry] Closed (%d): %s"), StatusCode, *Reason);
}

void ATelemetryReceiver::OnError(const FString& Error)
{
    UE_LOG(LogTemp, Error, TEXT("[Telemetry] Error: %s"), *Error);
}

static bool ParseVector(const TSharedPtr<FJsonObject>& Obj, const FString& Key, FVector& Out)
{
    const TArray<TSharedPtr<FJsonValue>>* Arr;
    if (!Obj->TryGetArrayField(Key, Arr) || Arr->Num() < 3)
    {
        return false;
    }
    Out.X = static_cast<float>((*Arr)[0]->AsNumber());
    Out.Y = static_cast<float>((*Arr)[1]->AsNumber());
    Out.Z = static_cast<float>((*Arr)[2]->AsNumber());
    return true;
}

static bool ParseQuat(const TSharedPtr<FJsonObject>& Obj, const FString& Key, FQuat& Out)
{
    const TArray<TSharedPtr<FJsonValue>>* Arr;
    if (!Obj->TryGetArrayField(Key, Arr) || Arr->Num() < 4)
    {
        return false;
    }
    Out.W = static_cast<float>((*Arr)[0]->AsNumber());
    Out.X = static_cast<float>((*Arr)[1]->AsNumber());
    Out.Y = static_cast<float>((*Arr)[2]->AsNumber());
    Out.Z = static_cast<float>((*Arr)[3]->AsNumber());
    return true;
}

void ATelemetryReceiver::OnMessage(const FString& Message)
{
    TSharedPtr<FJsonObject> Obj;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
    if (!FJsonSerializer::Deserialize(Reader, Obj) || !Obj.IsValid())
    {
        return;
    }

    FTelemetryState Parsed;
    ParseVector(Obj, TEXT("position"), Parsed.Position);
    ParseQuat(Obj, TEXT("quaternion"), Parsed.Orientation);
    ParseVector(Obj, TEXT("velocity"), Parsed.Velocity);
    ParseVector(Obj, TEXT("rates"), Parsed.Rates);
    ParseVector(Obj, TEXT("surfaces"), Parsed.Surfaces);

    Parsed.Throttle   = static_cast<float>(Obj->GetNumberField(TEXT("throttle")));
    Parsed.Soc        = static_cast<float>(Obj->GetNumberField(TEXT("soc")));
    Parsed.Voltage    = static_cast<float>(Obj->GetNumberField(TEXT("voltage")));
    Parsed.LoadFactor = static_cast<float>(Obj->GetNumberField(TEXT("load_factor")));
    Parsed.Heartbeat  = Obj->HasTypedField<EJson::Number>(TEXT("heartbeat"))
                            ? static_cast<int32>(Obj->GetNumberField(TEXT("heartbeat")))
                            : 0;
    Parsed.SimTime    = Obj->HasTypedField<EJson::Number>(TEXT("sim_time"))
                            ? static_cast<float>(Obj->GetNumberField(TEXT("sim_time")))
                            : 0.0f;
    Parsed.Mode       = Obj->GetStringField(TEXT("mode"));

    {
        FScopeLock Lock(&StateMutex);
        PendingState = Parsed;
        bHasPending = true;
    }
}
