CREATE MACRO sentence(x) AS upper(x[1]) || lower(x[2:]);
-- Star schema: dim_region, dim_item, dim_date, fact_cpi, fact_weight
CREATE TABLE dim_region AS
SELECT row_number() OVER (ORDER BY is_national DESC, region_name) AS region_id, region_name, region_short, is_national
FROM (
  SELECT DISTINCT
    geo_raw AS region_name,
    coalesce(nullif(regexp_extract(geo_raw, '\(([^)]+)\)', 1), ''), sentence(geo_raw)) AS region_short,
    geo_raw = 'Philippines' AS is_national
  FROM raw_cpi);

CREATE TABLE dim_item AS
SELECT row_number() OVER (ORDER BY item_code) AS item_id, item_code, item_name, item_code = '0' AS is_headline
FROM (SELECT DISTINCT split_part(item_raw, ' - ', 1) AS item_code,
             sentence(split_part(item_raw, ' - ', 2)) AS item_name FROM raw_cpi);

CREATE TABLE dim_date AS
SELECT CAST(strftime(d, '%Y%m') AS INTEGER) AS date_id, d AS month_start,
       year(d) AS year, month(d) AS month, strftime(d, '%b') AS month_name, quarter(d) AS quarter
FROM (SELECT DISTINCT date AS d FROM raw_cpi);

CREATE TABLE fact_cpi AS
SELECT r.region_id, i.item_id, CAST(strftime(c.date, '%Y%m') AS INTEGER) AS date_id,
       c.value AS cpi, y.value AS yoy_pct
FROM raw_cpi c
JOIN dim_region r ON c.geo_raw = r.region_name
JOIN dim_item i ON split_part(c.item_raw, ' - ', 1) = i.item_code
LEFT JOIN raw_yoy y ON y.geo_raw = c.geo_raw AND y.item_raw = c.item_raw AND y.date = c.date;

-- Raw weights are each cell's share of the NATIONAL basket (regions sum to 100).
-- weight_national = that raw share; weight_in_region = share within the region's own basket (sums to 100 per region).
CREATE TABLE fact_weight AS
SELECT r.region_id, i.item_id, w."Weights" AS weight_national,
       100.0 * w."Weights" / sum(CASE WHEN i.is_headline THEN w."Weights" END) OVER (PARTITION BY r.region_id) AS weight_in_region
FROM raw_weights w
JOIN dim_region r ON w."Geolocation" = r.region_name
JOIN dim_item i ON split_part(w."Commodity Description", ' - ', 1) = i.item_code;
