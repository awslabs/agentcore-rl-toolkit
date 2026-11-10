"""BIRD-Interact's a-Interact environment for one task: the actions, the budget, the user simulator and the tests.

Ported from bird-bench/BIRD-Interact@451fe2c (MIT), bird_interact_agent/batch_run_bird_interact/ (main.py,
action_handler.py, prompt_utils.py) and src/envs/ (test_case_utils/, user_simulator/), with the same rules:

- Budget: 3 + 3 + 2 x (critical and knowledge ambiguities) + patience; each action costs ACTION_COSTS. The action
  that exhausts it still runs, then only submit is accepted, and that submit ends the task.
- execute and submit run on a freshly reset database: this episode's own copy, recreated from <db>_template before
  each of them (upstream resets the shared database; a copy per episode lets episodes run concurrently).
- submit: phase 1 (the ambiguous query) earns 0.7, then the follow-up (phase 2) 0.3. Query tasks compare result
  sets with the gold SQL; Management tasks run the gold test cases. A failed submit returns the reason.
- ask: the two-stage user simulator (v2 prompts): the encoder maps the question to labeled / unlabeled /
  unanswerable, the decoder answers in the user's words. Upstream uses gemini-2.0-flash; here Claude Haiku 4.5 on
  Bedrock at temperature 0.
"""

import json
import re
import threading
import time
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

import boto3
import psycopg2
from botocore.config import Config
from sqlglot import tokenize
from sqlglot.tokens import TokenType

ACTION_COSTS = {
    "ask": 2,
    "submit": 3,
    "execute": 1,
    "get_schema": 1,
    "get_all_column_meanings": 1,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1,
}
WRONG_FORMAT_COST = 0.2  # a turn without an action
PHASE_REWARDS = {1: 0.7, 2: 0.3}
MAX_RESULT_WORDS = 500
MAX_ROWS = 10000
KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]
SIMULATOR = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_bedrock = boto3.client(
    "bedrock-runtime", region_name="us-west-2", config=Config(retries={"max_attempts": 8, "mode": "adaptive"})
)


class InfraError(Exception):
    """The database or the user simulator is unavailable: not the agent's fault."""


# --- SQL execution and the tests (src/envs/bird_interact_env/test_case_utils/) ---


def perform_query(query: str, conn):
    """One statement with a 60 s limit, committed; SELECT / WITH results capped at 10,000 rows."""
    cursor = conn.cursor()
    cursor.execute("SET statement_timeout = '60s';")
    try:
        cursor.execute(query)
        lower_q = query.strip().lower()
        conn.commit()
        if lower_q.startswith("select") or lower_q.startswith("with"):
            return cursor.fetchmany(MAX_ROWS + 1)[:MAX_ROWS]
        try:
            return cursor.fetchall()
        except psycopg2.ProgrammingError:
            return None
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def execute_queries(queries, db_name, conn=None, logger=None, section_title="", is_solution=True):
    """(the last query's result, error message or False, timed out) for a list of queries on one connection;
    the test cases call it by this name and signature."""
    query_result, execution_error, timeout_error = None, False, False
    for query in [queries] if isinstance(queries, str) else queries:
        try:
            query_result = perform_query(query, conn)
        except psycopg2.errors.QueryCanceled:
            timeout_error = timeout_error or is_solution
        except psycopg2.OperationalError as e:
            if e.pgcode is None:  # the connection itself failed
                raise InfraError(f"database: {e}") from None
            execution_error = str(e) if is_solution else execution_error
        except Exception as e:  # noqa: BLE001 - a failing query is the query's result
            execution_error = str(e) if is_solution else execution_error
    return query_result, execution_error, timeout_error


def process_decimals_recursive(item, decimal_places):
    quantizer = Decimal(1).scaleb(-decimal_places)
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    if isinstance(item, float):
        return round(item, decimal_places)
    if isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    if isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    return item


def preprocess_results(results, decimal_places=2):
    """Dates as YYYY-MM-DD, decimals rounded, unhashable values as sorted JSON."""
    processed = []
    for result in results:
        row = []
        for item in result:
            if isinstance(item, (date, datetime)):
                row.append(item.strftime("%Y-%m-%d"))
            else:
                item = process_decimals_recursive(item, decimal_places)
                row.append(json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item)
        processed.append(tuple(row))
    return processed


def remove_comments(sql_list):
    cleaned = []
    for sql in sql_list:
        no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
        cleaned.append(re.sub(r"\n\s*\n+", "\n", no_line).strip())
    return cleaned


def remove_distinct(sql_list):
    return [" ".join(t for t in q.split(" ") if t.lower() != "distinct") for q in sql_list]


def remove_round_functions(sql_string):
    """ROUND(x, n) -> x, nested ones included."""

    def find_matching_paren(text, start_pos):
        depth = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def find_first_arg_end(text, start_pos):
        depth = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                if depth == 0:
                    return i
                depth -= 1
            elif text[i] == "," and depth == 0:
                return i
        return len(text)

    result = sql_string
    pattern = re.compile(r"ROUND\s*\(", re.IGNORECASE)
    while match := pattern.search(result):
        open_paren = match.end() - 1
        first_arg_end = find_first_arg_end(result, open_paren + 1)
        close_paren = find_matching_paren(result, open_paren)
        if close_paren == -1:
            break
        result = result[: match.start()] + result[open_paren + 1 : first_arg_end].strip() + result[close_paren + 1 :]
    return result


def remove_round(sql_list):
    return [remove_round_functions(sql) for sql in sql_list]


def ex_base(pred_sqls, sol_sqls, db_name, conn, conditions=None):
    """1 if the two queries' result sets match (as lists when conditions["order"]), else 0."""
    if not pred_sqls or not sol_sqls:
        return 0
    predicted_res, pred_err, pred_to = execute_queries(pred_sqls, db_name, conn)
    ground_res, gt_err, gt_to = execute_queries(sol_sqls, db_name, conn)
    if any([pred_err, pred_to, gt_err, gt_to]):
        return 0
    predicted_res, ground_res = preprocess_results(predicted_res), preprocess_results(ground_res)
    if not predicted_res or not ground_res:
        return 0
    if conditions is not None and conditions.get("order", False):
        return 1 if predicted_res == ground_res else 0
    return 1 if set(predicted_res) == set(ground_res) else 0


def test_case_default(pred_sqls, sol_sqls, db_name, conn, conditions=None):
    pred_sqls = remove_round(remove_distinct(remove_comments(pred_sqls)))
    sol_sqls = remove_round(remove_distinct(remove_comments(sol_sqls)))
    result = ex_base(pred_sqls, sol_sqls, db_name, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result


# --- The user simulator (src/envs/user_simulator/, batch_run_bird_interact/prompt_utils.py) ---

_CLAUSES = {
    TokenType.SELECT: "SELECT",
    TokenType.FROM: "FROM",
    TokenType.WHERE: "WHERE",
    TokenType.HAVING: "HAVING",
    TokenType.GROUP_BY: "GROUP BY",
    TokenType.ORDER_BY: "ORDER BY",
    TokenType.LIMIT: "LIMIT",
    TokenType.OFFSET: "OFFSET",
    TokenType.JOIN: "JOIN",
    TokenType.STRAIGHT_JOIN: "STRAIGHT JOIN",
}


def segment_sql(sql: str) -> list[tuple[str, str]]:
    """(clause, text) slices of one statement, or its ;-separated statements if it does not tokenize."""
    try:
        starts = sorted(
            (t.start, _CLAUSES[t.token_type]) for t in tokenize(sql, read="postgres") if t.token_type in _CLAUSES
        )
        if not starts:
            return [("STATEMENT", sql.strip())]
        return [
            (name, sql[pos : starts[i + 1][0] if i + 1 < len(starts) else len(sql)].strip())
            for i, (pos, name) in enumerate(starts)
        ]
    except Exception:  # noqa: BLE001
        return [("STATEMENT", p + ";" if not p.endswith(";") else p) for p in (p.strip() for p in sql.split(";")) if p]


def sql_segments(sql) -> str:
    return "\n===\n".join(
        "".join(f"{clause}:\n{text}\n\n" for clause, text in segment_sql(s))
        for s in ([sql] if isinstance(sql, str) else sql)
    ).strip()


def simulate(prompt: str) -> str:
    try:
        r = _bedrock.converse(
            modelId=SIMULATOR,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 4096, "temperature": 0},
        )
    except Exception as e:  # noqa: BLE001
        raise InfraError(f"user simulator: {type(e).__name__}: {str(e)[:200]}") from None
    return "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b)


def between_s(response: str) -> str:
    """The text inside <s>...</s> (parse_encoder_response)."""
    response = response[: response.find("</s>")].strip() if "</s>" in response else response
    return response[response.find("<s>") :].replace("<s>", "").strip() if "<s>" in response else response


# --- One task ---


class Episode:
    def __init__(self, record: dict, store, patience: int, prompts: dict, copy: str):
        self.record, self.store, self.prompts, self.copy = record, store, prompts, copy
        self.used, self.lock = time.time(), threading.Lock()  # for the service
        amb = len(record.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
        amb += len(record.get("knowledge_ambiguity", []))
        self.total = self.remaining = 3 + 3 + amb * ACTION_COSTS["ask"] + patience
        self.phase, self.force_submit, self.done = 1, False, False
        self.phase_passed = {1: False, 2: False}
        self.phase1_sql = None
        self.db = record["selected_database"]

    @property
    def reward(self) -> float:
        return sum(PHASE_REWARDS[p] for p, ok in self.phase_passed.items() if ok)

    def opening(self) -> str:
        return (
            "Now, let's start with the user's question that may exist ambiguities and require external knowledge "
            f"understanding to solve.\n\nUser's Question: {self.record['amb_user_query']}\n\n[SYSTEM NOTE: You have a "
            f"total action budget of {self.total:.1f} units. Each action consumes budget. If the budget runs out, you "
            "must submit.]"
        )

    def note(self) -> str:
        if self.done:
            return f"\n\n[SYSTEM NOTE: Task finished. Total reward: {self.reward} points.]"
        return f"\n\n[SYSTEM NOTE: Remaining budget: {self.remaining:.1f}/{self.total:.1f}]" + (
            " Budget depleted. You MUST submit next." if self.force_submit else ""
        )

    def act(self, action: str, args: dict) -> str:
        """The observation of one action (None: a turn without one), with the budget note."""
        if self.done:
            return "[SYSTEM NOTE: Task finished.]"
        if self.force_submit and action != "submit":
            self.done = True
            return "[SYSTEM NOTE: Budget depleted. Agent failed to submit. Task failed.]"
        self.remaining -= ACTION_COSTS.get(action, WRONG_FORMAT_COST)
        if self.remaining <= 0 and not self.force_submit:
            self.force_submit = True
        if action is None:
            return "Error: no action. Call one of the tools: an Environment action or ask / submit." + self.note()
        if action == "ask":
            return self.ask(args["question"]) + self.note()
        if action == "submit":
            forced = self.force_submit
            out = self.submit(args["sql"])
            self.done = self.done or forced
            return out + self.note()
        return str(self.env_action(action, args)) + self.note()  # some column meanings are JSON objects

    # Environment actions (execute_env_action)
    def env_action(self, action: str, args: dict) -> str:
        info = self.store.db_info(self.db)
        if action == "execute":
            conn = self.store.fresh_copy(self)
            try:
                return format_result(conn, args["sql"])
            finally:
                conn.close()
        if action == "get_schema":
            return info["schema"]
        if action == "get_all_column_meanings":
            return json.dumps(info["column_meanings"], indent=2)
        if action == "get_column_meaning":
            key = f"{self.db}|{args['table_name'].lower()}|{args['column_name'].lower()}"
            return info["column_meanings"].get(key, "Column meaning not found")
        kb = self.knowledge()
        if action == "get_all_external_knowledge_names":
            return str(list(kb))
        if action == "get_knowledge_definition":
            k = kb.get(args["knowledge_name"])
            return (
                json.dumps({f: k[f] for f in KNOWLEDGE_VISIBLE_FIELDS if f in k}, indent=2)
                if k
                else ("Knowledge not found or not accessible.")
            )
        if action == "get_all_knowledge_definitions":
            return json.dumps([{f: k[f] for f in KNOWLEDGE_VISIBLE_FIELDS if f in k} for k in kb.values()], indent=2)
        raise ValueError(f"unknown action {action}")

    def knowledge(self) -> dict:
        """The knowledge base without the entries this task's knowledge ambiguities delete."""
        deleted = {
            a["deleted_knowledge"]
            for a in self.record.get("knowledge_ambiguity") or []
            if a.get("deleted_knowledge") is not None
        }
        return {name: k for name, k in self.store.db_info(self.db)["kb"].items() if k.get("id") not in deleted}

    # ask (encoder, then decoder)
    def ask(self, question: str) -> str:
        r = self.record
        follow = r.get("follow_up") or {}
        amb = {
            "user_query_ambiguity": r.get("user_query_ambiguity", {}),
            "knowledge_ambiguity": r.get("knowledge_ambiguity", []),
        }
        amb_json = json.dumps(amb if self.phase == 1 else {}, indent=2)
        gold = r.get("sol_sql", "") if self.phase == 1 else follow.get("sol_sql", "")
        clear = r.get("query", r.get("amb_user_query")) if self.phase == 1 else follow.get("query", "")
        schema, segments = self.store.db_info(self.db)["schema"], sql_segments(gold)
        fill = {
            "[[clarification_Q]]": question,
            "[[amb_json]]": amb_json,
            "[[SQL_Glot]]": segments,
            "[[DB_schema]]": schema,
        }
        encoder = self.prompts["encoder"]
        for k, v in fill.items():
            encoder = encoder.replace(k, v)
        action = between_s(simulate(encoder))
        decoder = self.prompts["decoder"].replace("[[clarification_Q]]", question).replace("[[Action]]", action)
        fill |= {"[[clear_query]]": clear, "[[GT_SQL]]": gold if isinstance(gold, str) else "\n".join(gold)}
        for k in ("[[amb_json]]", "[[clear_query]]", "[[GT_SQL]]", "[[SQL_Glot]]", "[[DB_schema]]"):
            decoder = decoder.replace(k, fill[k])
        return between_s(simulate(decoder))

    # submit (execute_submit_action, and the phase transition of main.py)
    def submit(self, sql: str) -> str:
        r, phase = self.record, self.phase
        spec = r if phase == 1 else (r.get("follow_up") or {})
        sol_sqls = spec.get("sol_sql", [])
        sol_sqls = sol_sqls if isinstance(sol_sqls, list) else [sol_sqls] if sol_sqls else []
        conn = self.store.fresh_copy(self)
        try:
            if phase == 2 and r.get("category") == "Management":
                with conn.cursor() as cur:
                    for s in [p for p in r.get("preprocess_sql") or [] if p]:
                        cur.execute(s)
                    if self.phase1_sql:
                        cur.execute(self.phase1_sql)
                        for s in [c for c in r.get("clean_up_sqls") or [] if c]:
                            cur.execute(s)
                conn.commit()
            passed, message = run_tests(sql, sol_sqls, spec, self.db, conn) if sol_sqls else (False, "no gold SQL")
        except InfraError:
            raise
        except Exception as e:  # noqa: BLE001 - upstream reports any error of the preparation as the observation
            conn.close()
            return f"Error processing submission: {e}"
        conn.close()
        if not passed:
            return f"Submitted SQL failed test case in Phase {phase}. Reason: {message} Please try again."
        self.phase_passed[phase] = True
        reward = PHASE_REWARDS[phase]
        if phase == 2:
            self.done = True
            return f"Phase 2 SQL Correct! (Reward: {reward} points). Task finished."
        self.phase1_sql = sql
        if not (r.get("follow_up") or {}).get("query"):
            self.done = True
            return f"Phase 1 SQL Correct! (Reward: {reward} points). No Phase 2. Task finished."
        if not self.force_submit:  # a forced submit ends the task, passed or not
            self.phase = 2
            return (
                f"Phase 1 SQL Correct! (Reward: {reward} points). Moving to Phase 2.\n\nNow, here's a follow-up "
                f"question: {r['follow_up']['query']}"
            )
        return f"Phase 1 SQL Correct! (Reward: {reward} points). Moving to Phase 2."


def run_tests(sql: str, sol_sqls: list, spec: dict, db_name: str, conn) -> tuple[bool, str]:
    pred_result, pred_err, pred_to = execute_queries(sql, db_name, conn)
    if pred_err:
        return False, f"Error executing submitted SQL: {pred_err}"
    if pred_to:
        return False, "Submitted SQL execution timed out"
    if spec.get("category", "Query") == "Query":
        try:
            test_case_default([sql], sol_sqls, db_name, conn, conditions=spec.get("conditions", {}))
            return True, "SQL passed default test case."
        except AssertionError as e:
            return False, f"Default test case failed: {e}"
        except InfraError:
            raise
        except Exception as e:  # noqa: BLE001
            return False, f"Error in default test case execution: {e}"
    exec_globals, exec_locals = {"pred_query_result": pred_result, "execute_queries": execute_queries}, {}
    for i, code in enumerate(c for c in spec.get("test_cases", []) if isinstance(c, str)):
        try:
            exec(code, exec_globals, exec_locals)  # the benchmark's own test case
            exec_locals["test_case"]([sql], sol_sqls, db_name, conn)
        except AssertionError as e:
            return False, f"SQL failed custom test cases: Test case {i + 1} assertion failed: {e}"
        except InfraError:
            raise
        except Exception as e:  # noqa: BLE001
            return False, f"SQL failed custom test cases: Error in custom test case {i + 1} execution: {e}"
    return True, "SQL passed all custom test cases."


def format_result(conn, sql: str) -> str:
    """execute's observation as upstream shows it: the rows (a list of tuples) cut at 500 words, or one value."""
    result, error, timeout = execute_queries(sql, "", conn)
    if error:
        return f"SQL execution error: {error}"
    if timeout:
        return "SQL execution timed out"
    if result is None:
        return "Query executed successfully."
    if not (isinstance(result, list) and all(isinstance(row, tuple) for row in result)):
        return f"Query executed successfully. Result: {result}"
    if not result:
        return "Query executed, empty result set."
    observation = f"Result: {result[0][0]}" if len(result) == 1 and len(result[0]) == 1 else str(result)
    words = observation.split()
    return " ".join(words[:MAX_RESULT_WORDS]) + "..." if len(words) > MAX_RESULT_WORDS else observation
