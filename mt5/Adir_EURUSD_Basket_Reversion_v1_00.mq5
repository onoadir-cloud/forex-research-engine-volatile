#property strict
#property version   "1.00"
#property description "Adir EURUSD Basket Reversion LONG-only EA"

// ============================================================================
// Demo / Forward Testing Notes:
// - Demo only
// - EURUSD only
// - M15 only
// - LONG only
// - No weekend carry
// - Basket/layering risk
// ============================================================================

#include <Trade/Trade.mqh>

input string InpSymbol = "EURUSD";
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M15;
input long InpMagicNumber = 26051901;

input double InpBaseLot = 0.01;
input double InpLotMultiplier = 1.2;
input int InpMaxLayers = 3;

input double InpMoveThresholdPips = 10.0;
input double InpLayerDistancePips = 10.0;
input double InpGroupTakeProfitPips = 8.0;
input double InpMaxAdversePips = 50.0;

input int InpMaxHoldBars = 80;

input int InpEntryHour1 = 16;
input int InpEntryHour2 = 17;
input int InpEntryHour3 = 18;

input double InpMaxSpreadPips = 1.5;

input bool InpForceFridayClose = true;
input int InpFridayCloseHour = 21;

input bool InpRequireHedgingAccount = true;
input bool InpPrintDebug = true;

CTrade g_trade;
datetime g_lastSignalBarTime = 0;
bool g_volumeStepWarned = false;
bool g_tradingEnabled = true;

// Explicit EURUSD-style pip size helper.
double PipSize()
{
   const int digits = (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS);
   if(digits == 5 || digits == 4)
      return 0.0001;

   if(InpPrintDebug)
      Print("WARNING: Unexpected digits for ", InpSymbol, " (", digits, "). Falling back to Point.");
   return SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
}

bool IsFridayAfterClose(const datetime when)
{
   if(!InpForceFridayClose)
      return false;

   MqlDateTime dt;
   TimeToStruct(when, dt);
   return (dt.day_of_week == 5 && dt.hour >= InpFridayCloseHour);
}

bool IsMondayWeekendRecoveryNeeded()
{
   double weightedAvg = 0.0;
   double totalVol = 0.0;
   datetime earliestOpen = 0;
   double lastEntryPrice = 0.0;
   if(!GetBasketStats(weightedAvg, totalVol, earliestOpen, lastEntryPrice))
      return false;

   MqlDateTime nowDt;
   TimeToStruct(TimeCurrent(), nowDt);
   if(nowDt.day_of_week != 1)
      return false;

   MqlDateTime mondayStartDt = nowDt;
   mondayStartDt.hour = 0;
   mondayStartDt.min = 0;
   mondayStartDt.sec = 0;
   const datetime mondayStart = StructToTime(mondayStartDt);

   return (earliestOpen > 0 && earliestOpen < mondayStart);
}

bool IsAllowedEntryHour(const datetime barTime)
{
   MqlDateTime dt;
   TimeToStruct(barTime, dt);
   const int h = dt.hour;
   return (h == InpEntryHour1 || h == InpEntryHour2 || h == InpEntryHour3);
}

bool SpreadOk(const string context)
{
   MqlTick tick;
   if(!SymbolInfoTick(InpSymbol, tick))
      return false;

   const double spreadPips = (tick.ask - tick.bid) / PipSize();
   if(spreadPips <= InpMaxSpreadPips)
      return true;

   if(InpPrintDebug)
      Print("Spread blocked ", context, ": ", DoubleToString(spreadPips, 2), " pips > max ", DoubleToString(InpMaxSpreadPips, 2));
   return false;
}

int CountBasketPositions()
{
   int count = 0;
   const int total = PositionsTotal();
   for(int i = 0; i < total; ++i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;

      if(PositionGetString(POSITION_SYMBOL) != InpSymbol)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;

      count++;
   }
   return count;
}

bool GetBasketStats(double &weightedAvg, double &totalVol, datetime &earliestOpen, double &lastEntryPrice)
{
   weightedAvg = 0.0;
   totalVol = 0.0;
   earliestOpen = 0;
   lastEntryPrice = 0.0;

   datetime latestOpen = 0;
   double weightedSum = 0.0;

   const int total = PositionsTotal();
   for(int i = 0; i < total; ++i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;

      if(PositionGetString(POSITION_SYMBOL) != InpSymbol)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;

      const double vol = PositionGetDouble(POSITION_VOLUME);
      const double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      const datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);

      weightedSum += openPrice * vol;
      totalVol += vol;

      if(earliestOpen == 0 || openTime < earliestOpen)
         earliestOpen = openTime;

      if(openTime >= latestOpen)
      {
         latestOpen = openTime;
         lastEntryPrice = openPrice;
      }
   }

   if(totalVol <= 0.0)
      return false;

   weightedAvg = weightedSum / totalVol;
   return true;
}

double NormalizeLots(const double rawLots)
{
   const double minLot = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN);
   const double maxLot = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MAX);
   const double step = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_STEP);

   double lots = MathMax(minLot, MathMin(maxLot, rawLots));
   if(step > 0.0)
      lots = MathRound(lots / step) * step;

   lots = MathMax(minLot, MathMin(maxLot, lots));

   // Final normalization for display precision.
   int stepDigits = 2;
   if(step > 0.0)
      stepDigits = (int)MathMax(0.0, -MathLog10(step));
   return NormalizeDouble(lots, stepDigits);
}

bool OpenBuyLayer(const int layerIndex, const string reason)
{
   const double rawLots = InpBaseLot * MathPow(InpLotMultiplier, layerIndex);
   const double normalizedLots = NormalizeLots(rawLots);

   if(layerIndex > 0)
   {
      const double prevLots = NormalizeLots(InpBaseLot * MathPow(InpLotMultiplier, layerIndex - 1));
      const double step = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_STEP);
      if(!g_volumeStepWarned && step > 0.0 && MathAbs(prevLots - normalizedLots) < step * 0.5)
      {
         g_volumeStepWarned = true;
         if(InpPrintDebug)
            Print("WARNING: volume step may neutralize lot multiplier. Consider larger BaseLot for multiplier testing.");
      }
   }

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   const bool ok = g_trade.Buy(normalizedLots, InpSymbol, 0.0, 0.0, 0.0, reason);
   if(InpPrintDebug)
   {
      Print("Open BUY ", reason,
            " layer=", layerIndex,
            " rawLots=", DoubleToString(rawLots, 4),
            " lots=", DoubleToString(normalizedLots, 4),
            " result=", (ok ? "OK" : "FAIL"),
            " retcode=", g_trade.ResultRetcode(),
            " comment=", g_trade.ResultComment());
   }
   return ok;
}

void CloseAllBasketPositions(const string reason)
{
   bool any = false;
   const int total = PositionsTotal();
   for(int i = total - 1; i >= 0; --i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;

      if(PositionGetString(POSITION_SYMBOL) != InpSymbol)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;

      any = true;
      const bool ok = g_trade.PositionClose(ticket);
      if(InpPrintDebug)
      {
         Print("Close ticket=", ticket,
               " reason=", reason,
               " result=", (ok ? "OK" : "FAIL"),
               " retcode=", g_trade.ResultRetcode(),
               " comment=", g_trade.ResultComment());
      }
   }

   if(any && InpPrintDebug)
      Print("Basket closed. reason=", reason);
}

int OnInit()
{
   Print("Init Adir_EURUSD_Basket_Reversion_v1_00 - demo forward-test EA");

   if(!SymbolInfoInteger(InpSymbol, SYMBOL_EXIST))
   {
      Print("ERROR: Symbol does not exist: ", InpSymbol);
      return(INIT_FAILED);
   }

   if(!SymbolSelect(InpSymbol, true))
   {
      Print("ERROR: Failed to select symbol: ", InpSymbol);
      return(INIT_FAILED);
   }

   if(InpTimeframe != PERIOD_M15)
      Print("WARNING: InpTimeframe is not M15. Strategy is validated for M15 only.");

   if(_Period != InpTimeframe)
      Print("WARNING: Chart timeframe differs from InpTimeframe. EA logic uses InpTimeframe data.");

   if(InpRequireHedgingAccount)
   {
      const ENUM_ACCOUNT_MARGIN_MODE mm = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
      if(mm != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      {
         Print("ERROR: Hedging account required. ACCOUNT_MARGIN_MODE=", (int)mm,
               ". Trading disabled because netting may merge layers.");
         g_tradingEnabled = false;
      }
      else
      {
         Print("Hedging account check: OK");
      }
   }
   else if(InpPrintDebug)
   {
      Print("Hedging account check bypassed by input.");
   }

   return(INIT_SUCCEEDED);
}

void OnTick()
{
   if(!g_tradingEnabled)
      return;

   MqlTick tick;
   if(!SymbolInfoTick(InpSymbol, tick))
      return;

   double weightedAvg = 0.0;
   double totalVol = 0.0;
   datetime firstOpen = 0;
   double lastEntryPrice = 0.0;
   const bool basketOpen = GetBasketStats(weightedAvg, totalVol, firstOpen, lastEntryPrice);
   bool weekendRecoveryClosedThisTick = false;

   // Forced Friday close check (applies whenever basket is open).
   if(basketOpen && IsFridayAfterClose(TimeCurrent()))
   {
      if(InpPrintDebug)
         Print("Friday close condition met. Closing basket.");
      CloseAllBasketPositions("FORCED_FRIDAY_CLOSE");
      return;
   }

   // Monday emergency weekend recovery close.
   if(basketOpen && IsMondayWeekendRecoveryNeeded())
   {
      Print("FORCED_WEEKEND_RECOVERY_CLOSE: basket was carried over weekend; closing immediately.");
      CloseAllBasketPositions("FORCED_WEEKEND_RECOVERY_CLOSE");
      weekendRecoveryClosedThisTick = true;
      return;
   }

   // Basket exit logic.
   if(basketOpen)
   {
      const double pip = PipSize();

      if(tick.bid >= weightedAvg + InpGroupTakeProfitPips * pip)
      {
         CloseAllBasketPositions("GROUP_TP");
         return;
      }

      if(tick.bid <= weightedAvg - InpMaxAdversePips * pip)
      {
         CloseAllBasketPositions("MAX_ADVERSE");
         return;
      }

      if(firstOpen > 0)
      {
         const int barsSinceFirst = iBarShift(InpSymbol, InpTimeframe, firstOpen, false);
         if(barsSinceFirst >= 0 && barsSinceFirst >= InpMaxHoldBars)
         {
            CloseAllBasketPositions("MAX_HOLD");
            return;
         }
      }

      // Layering logic.
      const int layers = CountBasketPositions();
      if(layers < InpMaxLayers)
      {
         if(!IsFridayAfterClose(TimeCurrent()) && !weekendRecoveryClosedThisTick)
         {
            const double trigger = lastEntryPrice - InpLayerDistancePips * pip;
            if(tick.bid <= trigger)
            {
               if(SpreadOk("layer"))
                  OpenBuyLayer(layers, "LAYER");
            }
         }
      }
      return;
   }

   // New-basket entry is evaluated only once per newly closed bar.
   const datetime signalBarTime = iTime(InpSymbol, InpTimeframe, 1);
   if(signalBarTime <= 0 || signalBarTime == g_lastSignalBarTime)
      return;

   g_lastSignalBarTime = signalBarTime;

   if(InpTimeframe != PERIOD_M15)
      return;

   if(IsFridayAfterClose(TimeCurrent()) || weekendRecoveryClosedThisTick)
      return;

   if(Bars(InpSymbol, InpTimeframe) < 18)
      return;

   if(!IsAllowedEntryHour(signalBarTime))
      return;

   const double signalClose = iClose(InpSymbol, InpTimeframe, 1);
   const double anchorClose = iClose(InpSymbol, InpTimeframe, 17);
   if(signalClose == 0.0 || anchorClose == 0.0)
      return;

   const double threshold = anchorClose - InpMoveThresholdPips * PipSize();
   if(signalClose > threshold)
      return;

   if(!SpreadOk("entry"))
      return;

   if(OpenBuyLayer(0, "NEW_BASKET"))
   {
      if(InpPrintDebug)
         Print("New basket opened.");
   }
}
