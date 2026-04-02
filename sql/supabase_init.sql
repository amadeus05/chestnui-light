-- Инициализация схемы для paper/live execution в Supabase (PostgreSQL)
-- Запустить целиком в Supabase Dashboard -> SQL Editor.

CREATE TABLE IF NOT EXISTS execution_trades (
    id SERIAL PRIMARY KEY,
    execution_type TEXT NOT NULL CHECK (execution_type IN ('paper', 'live')),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    exchange_code TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction INTEGER NOT NULL CHECK (direction IN (1, -1)),
    timeframe_signal TEXT NOT NULL DEFAULT '1h',
    signal_bar_open_ms BIGINT NOT NULL,
    entry_bar_open_ms BIGINT,
    entry_ts_ms BIGINT NOT NULL,
    entry_price REAL NOT NULL,
    entry_notional REAL NOT NULL,
    stop_pct REAL NOT NULL,
    take_pct REAL NOT NULL,
    p_long REAL,
    p_short REAL,
    signal_gap REAL,
    model_name TEXT,
    last_1m_scan_open_ms BIGINT,
    exit_ts_ms BIGINT,
    exit_price REAL,
    exit_reason TEXT,
    pnl_pct REAL,
    pnl_quote REAL,
    fees_quote REAL,
    meta_json TEXT,
    created_at_ms BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exec_trades_type_status_time
    ON execution_trades (execution_type, status, entry_ts_ms);

CREATE INDEX IF NOT EXISTS idx_exec_trades_symbol_entry
    ON execution_trades (symbol, entry_ts_ms);

CREATE INDEX IF NOT EXISTS idx_exec_trades_exit_time
    ON execution_trades (execution_type, exit_ts_ms);

CREATE TABLE IF NOT EXISTS execution_engine_state (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

-- RPC: количество SL за текущие сутки (UTC)
CREATE OR REPLACE FUNCTION count_sl_today(
    p_execution_type TEXT,
    p_symbol TEXT,
    p_day_start_ms BIGINT,
    p_now_ms BIGINT
) RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*) FROM execution_trades
        WHERE execution_type = p_execution_type
          AND status = 'closed'
          AND symbol = p_symbol
          AND exit_reason = 'SL'
          AND exit_ts_ms >= p_day_start_ms
          AND exit_ts_ms <= p_now_ms
    );
END;
$$ LANGUAGE plpgsql;

-- RPC: время последнего SL по символу
CREATE OR REPLACE FUNCTION last_sl_exit_ms(
    p_execution_type TEXT,
    p_symbol TEXT
) RETURNS BIGINT AS $$
BEGIN
    RETURN (
        SELECT MAX(exit_ts_ms) FROM execution_trades
        WHERE execution_type = p_execution_type
          AND status = 'closed'
          AND symbol = p_symbol
          AND exit_reason = 'SL'
    );
END;
$$ LANGUAGE plpgsql;

-- RPC: сумма закрытого PnL
CREATE OR REPLACE FUNCTION sum_closed_pnl(
    p_execution_type TEXT
) RETURNS REAL AS $$
BEGIN
    RETURN COALESCE(
        (SELECT SUM(pnl_quote) FROM execution_trades
         WHERE execution_type = p_execution_type AND status = 'closed'),
        0
    );
END;
$$ LANGUAGE plpgsql;

-- RPC: суммарный notional открытых позиций
CREATE OR REPLACE FUNCTION sum_open_notional(
    p_execution_type TEXT
) RETURNS REAL AS $$
BEGIN
    RETURN COALESCE(
        (SELECT SUM(entry_notional) FROM execution_trades
         WHERE execution_type = p_execution_type AND status = 'open'),
        0
    );
END;
$$ LANGUAGE plpgsql;
