-- One row per region-month: today's price signals + next month's inflation (the target).
CREATE OR REPLACE VIEW features AS
WITH p AS (
  SELECT f.region_id, d.month_start, i.item_code, f.cpi, f.yoy_pct
  FROM fact_cpi f JOIN dim_item i USING (item_id) JOIN dim_date d USING (date_id)
), w AS (
  SELECT region_id, month_start,
    max(CASE WHEN item_code = '0'  THEN cpi END)     AS cpi_all,  max(CASE WHEN item_code = '0'  THEN yoy_pct END) AS yoy_all,
    max(CASE WHEN item_code = '01' THEN cpi END)     AS cpi_food, max(CASE WHEN item_code = '01' THEN yoy_pct END) AS yoy_food,
    max(CASE WHEN item_code = '04' THEN cpi END)     AS cpi_hous, max(CASE WHEN item_code = '04' THEN yoy_pct END) AS yoy_hous,
    max(CASE WHEN item_code = '07' THEN cpi END)     AS cpi_trans, max(CASE WHEN item_code = '07' THEN yoy_pct END) AS yoy_trans
  FROM p GROUP BY region_id, month_start
), l AS (
  SELECT *,
    lag(yoy_all, 1) OVER s AS yoy_l1, lag(yoy_all, 2) OVER s AS yoy_l2, lag(yoy_all, 3) OVER s AS yoy_l3,
    100 * (cpi_all   / lag(cpi_all, 1)   OVER s - 1) AS mom_all,
    100 * (cpi_all   / lag(cpi_all, 3)   OVER s - 1) AS mom3_all,
    100 * (cpi_food  / lag(cpi_food, 1)  OVER s - 1) AS mom_food,
    100 * (cpi_hous  / lag(cpi_hous, 1)  OVER s - 1) AS mom_hous,
    100 * (cpi_trans / lag(cpi_trans, 1) OVER s - 1) AS mom_trans,
    lead(yoy_all, 1) OVER s AS target_yoy
  FROM w WINDOW s AS (PARTITION BY region_id ORDER BY month_start)
), nat AS (
  SELECT l.month_start, l.yoy_all AS nat_yoy FROM l JOIN dim_region r USING (region_id) WHERE r.is_national
)
SELECT l.*, nat.nat_yoy, l.yoy_all - nat.nat_yoy AS gap_vs_nat, month(l.month_start) AS month_num,
       r.region_name, r.region_short, r.is_national
FROM l JOIN nat USING (month_start) JOIN dim_region r USING (region_id);
