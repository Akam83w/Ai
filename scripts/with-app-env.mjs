#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Minimal env wrapper: run the remaining CLI args as a child process.
// The original Grok app-builder helper is not in this repository.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("usage: node scripts/with-app-env.mjs <command> [args...]");
  process.exit(1);
}

const child = spawn(args[0], args.slice(1), {
  stdio: "inherit",
  shell: false,
  env: process.env,
  cwd: path.resolve(path.dirname(fileURLToPath(import.meta.url)), ".."),
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 1);
});
