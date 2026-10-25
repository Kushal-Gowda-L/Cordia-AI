import type { TooltipProps } from "recharts";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PATIENT_VIEW } from "../../patient/patientTwin";
import { useHeartStore, type RiskPoint } from "../../state/store";

const DISEASE_LEVELS = ["Mild", "Moderate", "Severe", "Critical"];
const AXIS = "#5b6b7c";
const GRID = "#e3e8ee";
const PRIMARY = "#1f5fae";

function HistoryTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0].payload as RiskPoint;
  return (
    <div className="chart-tooltip">
      <b>{point.risk}%</b>
      <div>{point.label}</div>
    </div>
  );
}

export function RiskCharts() {
  const prediction = useHeartStore((s) => s.prediction);
  const recordTwin = useHeartStore((s) => s.recordTwin);
  const history = useHeartStore((s) => s.riskHistory);
  const hasPatient = useHeartStore((s) => s.patient !== null);
  if (!prediction) return null;

  const baseline = recordTwin?.prediction;
  const baselineName = PATIENT_VIEW ? "Your results" : hasPatient ? "Record" : "Starting values";
  // Step 2 of the reading: disease levels rescaled to add up to 100% among disease cases.
  const bars = DISEASE_LEVELS.map((name) => ({
    name,
    baseline: baseline?.reading.severity_if_disease[name] ?? 0,
    now: prediction.reading.severity_if_disease[name] ?? 0,
  }));
  const { zones, decision_point: cut } = prediction.reading;

  return (
    <>
      {!PATIENT_VIEW && (
      <div className="panel">
        <div className="panel-title">
          <span>If heart disease is present, how severe?</span>
          <span className="panel-subtitle">
            The four disease levels rescaled to add up to 100% (each level&rsquo;s chance divided by the total chance
            of disease).
          </span>
        </div>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={bars} margin={{ top: 4, right: 6, left: -18, bottom: 0 }} barGap={2}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: AXIS }} stroke={GRID} interval={0} />
            <YAxis tick={{ fontSize: 11, fill: AXIS }} stroke={GRID} domain={[0, 100]} unit="%" />
            <Tooltip formatter={(v) => `${v}%`} cursor={{ fill: "#f1f4f8" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="baseline" name={baselineName} fill="#b9c5d3" isAnimationActive={false} />
            <Bar dataKey="now" name="Now" fill={PRIMARY} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      )}

      <div className="panel">
        <div className="panel-title">
          <span>Chance of heart disease as details change</span>
          <span className="panel-subtitle">
            One point per change &mdash; hover for what changed. Shading: unlikely / borderline / likely zones; dark
            line: the model&rsquo;s {cut}% decision point; dashed line: {baselineName.toLowerCase()}.
          </span>
        </div>
        <ResponsiveContainer width="100%" height={190}>
          <LineChart data={history} margin={{ top: 8, right: 10, left: -18, bottom: 0 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis dataKey="step" tick={{ fontSize: 11, fill: AXIS }} stroke={GRID} allowDecimals={false} />
            <YAxis tick={{ fontSize: 11, fill: AXIS }} stroke={GRID} domain={[0, 100]} unit="%" />
            <Tooltip content={<HistoryTooltip />} />
            <ReferenceArea y1={0} y2={zones.unlikely_below} fill="#2e7d4f" fillOpacity={0.07} />
            <ReferenceArea y1={zones.unlikely_below} y2={zones.likely_from} fill="#d99a1e" fillOpacity={0.1} />
            <ReferenceArea y1={zones.likely_from} y2={100} fill="#c0392b" fillOpacity={0.07} />
            <ReferenceLine y={cut} stroke="#5b6b7c" strokeWidth={1.5} />
            {baseline && <ReferenceLine y={baseline.risk_percent} stroke="#8a97a6" strokeDasharray="4 4" />}
            <Line
              type="monotone"
              dataKey="risk"
              stroke={PRIMARY}
              strokeWidth={2}
              dot={{ r: 3 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
