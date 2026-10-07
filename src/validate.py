"""Great Expectations quality gate and portable HTML evidence."""
import html
from pathlib import Path
import great_expectations as gx
from src.prepare import write_json, require

def run_validation(frame, output):
    context = gx.get_context(mode='ephemeral')
    context.enable_analytics(False)
    source = context.data_sources.add_pandas('m5_prepared')
    asset = source.add_dataframe_asset(name='daily_cohort')
    definition = asset.add_batch_definition_whole_dataframe('all_rows')
    batch = definition.get_batch(batch_parameters={'dataframe':frame})
    suite = gx.ExpectationSuite(name='m5_daily_quality')
    suite.add_expectation(gx.expectations.ExpectTableRowCountToEqual(value=78600))
    suite.add_expectation(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=['id','date']))
    for c in ['id','item_id','store_id','date','units','split']:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=c))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column='units', min_value=0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column='split', value_set=['train','validation','test']))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column='cat_id', value_set=['HOUSEHOLD']))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column='zero_sales_cause', value_set=['unknown','not_zero']))
    result = batch.validate(suite)
    output = Path(output)
    write_json(output/'gx_suite.json', suite.to_json_dict())
    write_json(output/'gx_validation.json', result.to_json_dict())
    rows = []
    for r in result.results:
        config = r.expectation_config.to_json_dict()
        rows.append('<tr><td>'+html.escape(config.get('type',''))+'</td><td>'+html.escape(str(config.get('kwargs',{})))+'</td><td>'+('PASS' if r.success else 'FAIL')+'</td></tr>')
    (output/'gx_report.html').write_text('<!doctype html><meta charset="utf-8"><title>M5 Great Expectations results</title><style>body{font:20px Arial;margin:40px;color:#173245}h1{font-size:32px}table{border-collapse:collapse;width:100%}td,th{padding:14px;border-bottom:1px solid #ccd6dc;text-align:left}th{background:#edf3f5}</style><h1>Great Expectations validation</h1><p>Actual M5 run: 78,600 daily records. Suite: m5_daily_quality</p><p>Overall result: '+('PASS' if result.success else 'FAIL')+'</p><table><tr><th>Expectation</th><th>Parameters</th><th>Result</th></tr>'+''.join(rows)+'</table>',encoding='utf-8')
    require(result.success, 'Great Expectations quality gate failed')
    return result.statistics
