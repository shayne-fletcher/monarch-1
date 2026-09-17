/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod bootstrap;
mod development_identity;

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use bootstrap::JoinToken;
use chrysalis_sqlite::Replica;
use chrysalis_sqlite::TableSchema;
use clap::Parser;
use clap::Subcommand;
use libsql::Builder;
use libsql::Connection;
use libsql::Value;

const CREATE_CHALKBOARD: &str = "
    CREATE TABLE IF NOT EXISTS chalkboard (
        id TEXT PRIMARY KEY NOT NULL,
        author TEXT NOT NULL,
        message TEXT NOT NULL
    )
";

const UPSERT_NOTE: &str = "
    INSERT INTO chalkboard (id, author, message)
    VALUES (?1, ?2, ?3)
    ON CONFLICT(id) DO UPDATE SET
        author = excluded.author,
        message = excluded.message
";

const SELECT_NOTES: &str = "
    SELECT id, author, message
    FROM chalkboard
    ORDER BY id
";

struct Note {
    id: String,
    author: String,
    message: String,
}

#[derive(Debug, Parser)]
#[command(name = "shared-chalkboard")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Initialize a local chalkboard database.
    Init { database: PathBuf },

    /// Insert or update one note in a local chalkboard database.
    Write {
        database: PathBuf,
        note_id: String,
        author: String,
        message: String,
    },

    /// Display the notes in a local chalkboard database.
    Show { database: PathBuf },

    /// Synchronize a local chalkboard database over Chrysalis.
    Sync {
        #[arg(long)]
        join: Option<JoinToken>,
        database: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Init { database } => init(&database).await,
        Command::Write {
            database,
            note_id,
            author,
            message,
        } => {
            let note = Note {
                id: note_id,
                author,
                message,
            };
            write(&database, note).await
        }
        Command::Show { database } => show(&database).await,
        Command::Sync { join, database } => sync(&database, join).await,
    }
}

fn chalkboard_schema() -> Result<TableSchema> {
    Ok(TableSchema::try_new(
        "chalkboard",
        CREATE_CHALKBOARD,
        &["id", "author", "message"],
        &["id"],
    )?)
}

async fn init(path: &Path) -> Result<()> {
    let connection = open_sqlite(path).await?;
    Replica::new(connection, [chalkboard_schema()?])
        .await
        .with_context(|| format!("initialize chalkboard in {}", path.display()))?;
    println!("initialized chalkboard in {}", path.display());
    Ok(())
}

async fn write(path: &Path, note: Note) -> Result<()> {
    let connection = open_sqlite(path).await?;
    let replica = Replica::new(connection, [chalkboard_schema()?])
        .await
        .with_context(|| format!("open chalkboard in {}", path.display()))?;
    let mut transaction = replica.transaction().await?;
    transaction
        .execute(
            UPSERT_NOTE,
            vec![
                Value::Text(note.id.clone()),
                Value::Text(note.author),
                Value::Text(note.message),
            ],
        )
        .await
        .with_context(|| format!("write note {} to {}", note.id, path.display()))?;
    transaction
        .capture_upsert("chalkboard", vec![Value::Text(note.id.clone())])
        .await?;
    transaction.commit().await?;
    println!("wrote note {} to {}", note.id, path.display());
    Ok(())
}

async fn show(path: &Path) -> Result<()> {
    let connection = open_sqlite(path).await?;
    let mut rows = connection
        .query(SELECT_NOTES, ())
        .await
        .with_context(|| format!("read chalkboard from {}", path.display()))?;

    println!("chalkboard {}", path.display());
    while let Some(row) = rows.next().await? {
        let id: String = row.get(0)?;
        let author: String = row.get(1)?;
        let message: String = row.get(2)?;
        println!("{id} | {author} | {message}");
    }
    Ok(())
}

async fn sync(path: &Path, join: Option<JoinToken>) -> Result<()> {
    let connection = open_sqlite(path).await?;
    let replica = Replica::new(connection, [chalkboard_schema()?])
        .await
        .with_context(|| format!("initialize replicated chalkboard in {}", path.display()))?;
    bootstrap::run(replica, join).await
}

async fn open_sqlite(path: &Path) -> Result<Connection> {
    let path = path
        .to_str()
        .with_context(|| format!("database path is not UTF-8: {}", path.display()))?;
    let database = Builder::new_local(path)
        .build()
        .await
        .with_context(|| format!("open SQLite database {path}"))?;
    let connection = database
        .connect()
        .with_context(|| format!("connect to SQLite database {path}"))?;
    Ok(connection)
}
