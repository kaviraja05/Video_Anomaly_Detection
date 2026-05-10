import React, { useEffect, useState } from 'react';
import { getUserHistory } from '../api/api';
import { Activity, Play, ShieldAlert, ShieldCheck } from 'lucide-react';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getUserHistory().then(data => {
      setHistory(data.history || []);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setLoading(false);
    });
  }, []);

  return (
    <div className="w-full max-w-5xl mx-auto">
      <div className="flex items-center gap-3 mb-8">
        <div className="p-2 bg-blue-500/20 rounded-xl text-blue-400 border border-blue-500/30">
          <Activity size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-theme-text">Analysis History</h2>
          <p className="text-theme-text opacity-60 text-sm">View your previously uploaded video results</p>
        </div>
      </div>
      
      {loading ? (
        <div className="text-center py-12 text-theme-text opacity-70 animate-pulse">Loading history...</div>
      ) : (
        <div className="bg-theme-card rounded-xl border border-theme-border overflow-hidden shadow-xl backdrop-blur-sm">
          {history.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="bg-theme-input text-theme-text opacity-70 text-xs uppercase tracking-wider">
                  <tr>
                    <th className="px-6 py-4 font-medium">Video File</th>
                    <th className="px-6 py-4 font-medium">Result</th>
                    <th className="px-6 py-4 font-medium">Anomaly Score</th>
                    <th className="px-6 py-4 font-medium">Segments</th>
                    <th className="px-6 py-4 font-medium">Date Analyzed</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-theme-border">
                  {history.map((record, idx) => (
                    <tr key={idx} className="hover:bg-theme-card/30 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-theme-input rounded-lg text-theme-text opacity-70">
                            <Play size={16} />
                          </div>
                          <span className="text-theme-text font-medium">{record.video_name}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${record.status === 'anomaly' ? 'bg-red-500/10 text-red-400 border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'}`}>
                          {record.status === 'anomaly' ? <ShieldAlert size={14} /> : <ShieldCheck size={14} />}
                          {record.status?.toUpperCase() || 'UNKNOWN'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-theme-input rounded-full overflow-hidden">
                            <div className={`h-full ${record.anomaly_score > 0.5 ? 'bg-red-500' : 'bg-emerald-500'}`} style={{ width: `${record.anomaly_score * 100}%` }}></div>
                          </div>
                          <span className={`${record.anomaly_score > 0.5 ? 'text-red-400' : 'text-emerald-400'} font-medium text-sm`}>
                            {(record.anomaly_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-theme-text opacity-70 text-sm">
                        {record.segments?.length || 0}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-theme-text opacity-60 text-sm">
                        {new Date(record.created_at.endsWith('Z') ? record.created_at : record.created_at + 'Z').toLocaleString(undefined, {
                          month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                        })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-16 text-center">
              <div className="inline-flex justify-center items-center w-16 h-16 rounded-full bg-theme-input text-theme-text opacity-50 mb-4">
                <Activity size={32} />
              </div>
              <h3 className="text-lg font-medium text-theme-text opacity-90 mb-1">No analysis history</h3>
              <p className="text-theme-text opacity-60 max-w-sm mx-auto">Upload and analyze your first video to see your history logs appear here.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default History;
