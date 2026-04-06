-- Drop existing tables if they exist (order matters due to foreign keys)
DROP TABLE IF EXISTS checkpoint_writes;
DROP TABLE IF EXISTS checkpoints;


-- Global Variables
SET GLOBAL max_allowed_packet = 67108864;  -- 64MB
SET GLOBAL wait_timeout = 28800;
SET GLOBAL interactive_timeout = 28800;

-- Create the database
CREATE DATABASE IF NOT EXISTS langgraph_chatbot
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
    
-- Use the langgraph_chatbot database
USE langgraph_chatbot;

-- Checkpoints table (LangGraph conversation state)
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id       VARCHAR(128)    NOT NULL,
    checkpoint_ns   VARCHAR(128)    NOT NULL DEFAULT '',
    checkpoint_id   VARCHAR(128)    NOT NULL,
    parent_id       VARCHAR(128)    NULL,
    checkpoint      LONGBLOB        NOT NULL,
    metadata        LONGBLOB        NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Checkpoint writes table (intermediate LangGraph state)
-- FIXED: Added surrogate key to avoid 3072 byte key length limit
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    id              BIGINT          NOT NULL AUTO_INCREMENT,
    thread_id       VARCHAR(128)    NOT NULL,
    checkpoint_ns   VARCHAR(128)    NOT NULL DEFAULT '',
    checkpoint_id   VARCHAR(128)    NOT NULL,
    task_id         VARCHAR(128)    NOT NULL,
    idx             INT             NOT NULL,
    channel         VARCHAR(128)    NOT NULL,
    type            VARCHAR(128)    NULL,
    value           LONGBLOB        NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_checkpoint (thread_id, checkpoint_ns, checkpoint_id, task_id, idx),
    INDEX idx_thread (thread_id),
    INDEX idx_checkpoint (checkpoint_id),
    INDEX idx_task (task_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



CREATE TABLE IF NOT EXISTS threads (
    thread_id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
);


-- Clear contents of both tables
TRUNCATE TABLE checkpoints;
TRUNCATE TABLE checkpoint_writes;
TRUNCATE TABLE threads;

-- Verify table structures
SHOW TABLES;

-- Check structure of checkpoint_writes
DESCRIBE checkpoint_writes;