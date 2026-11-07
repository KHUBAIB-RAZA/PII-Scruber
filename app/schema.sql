CREATE TABLE statements (
    id VARCHAR(36) PRIMARY KEY,
    source_names TEXT NOT NULL,
    scrubbed_text TEXT NOT NULL,
    analysis TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    statement_id VARCHAR(36) NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX chat_messages_statement_id_idx ON chat_messages(statement_id);
