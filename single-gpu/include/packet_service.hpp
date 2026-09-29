#pragma once
#include "common.hpp"
namespace tqr {
// Packet count is a launch grid, not serial work: interpolate the observed batch curve; extrapolate
// its waves using queried residency. This interpolation is predictive and has no duration or
// physical-error bound.
class PacketServices {
 struct Point{int rows,h,count,threads;double time;};
 struct Residency{int rows,h,threads,blocks;};
 std::map<std::string,std::vector<Point>>points;
 std::map<std::string,std::vector<Residency>>residency;
 int sms=1;
 int capacity(const std::string&op,int rows,int h,int threads)const{
  auto it=residency.find(op);int blocks=1;double nearest=INFINITY;
  if(it!=residency.end())for(auto&r:it->second)if(r.threads==threads){double d=std::abs(std::log(double(rows)/r.rows))+2*std::abs(std::log(double(h)/r.h));if(d<nearest){nearest=d;blocks=r.blocks;}}
  return sms*std::max(1,blocks);
 }
public:
 explicit PacketServices(const json&p){sms=p.at("hardware").value("sm_count",1);
  for(auto&s:p.at("tiled_samples")){std::string op=s["op"];if(op=="GE"||op=="TT"||op=="GE_shared"||op=="TT_shared")points[op].push_back({s["rows"],s["h"],s["batch"],s["threads"],s["median_s"]});}
  for(auto name:{"shared_panel_resources","global_panel_resources"})if(p.contains(name))for(auto&r:p[name]){std::string op=r.at("kind")==0?"GE":"TT";if(std::string(name)=="shared_panel_resources")op+="_shared";residency[op].push_back({r["rows"],r["h"],r["threads"],r["active_blocks_per_SM"]});}
 }
 bool supports(const std::string&op)const{return points.count(op);}
 double get(const std::string&op,int rows,int h,int count,int threads)const{
  const auto&all=points.at(op);int nearest_h=0;double distance=INFINITY;
  for(auto&s:all)if(s.threads==threads){double d=std::abs(std::log(double(h)/s.h));if(d<distance){distance=d;nearest_h=s.h;}}
  if(!nearest_h)return INFINITY;
  std::map<int,std::map<int,double>>curves;for(auto&s:all)if(s.threads==threads&&s.h==nearest_h)curves[s.rows][s.count]=s.time;
  std::map<int,double>row_times;
  for(auto&[r,curve]:curves){auto hi=curve.lower_bound(count);double t;
   if(hi==curve.begin())t=hi->second;
   else if(hi==curve.end()){auto last=std::prev(hi);int cap=capacity(op,r,nearest_h,threads);t=last->second*double(ceildiv(count,cap))/ceildiv(last->first,cap);}
   else if(hi->first==count)t=hi->second;
   else{auto lo=std::prev(hi);double w=std::log(double(count)/lo->first)/std::log(double(hi->first)/lo->first);t=lo->second*(1-w)+hi->second*w;}
   row_times[r]=t;
  }
  auto hi=row_times.lower_bound(rows);double t;
  if(hi==row_times.begin())t=hi->second;
  else if(hi==row_times.end()){auto last=std::prev(hi);t=last->second*double(rows)/last->first;}
  else if(hi->first==rows)t=hi->second;
  else{auto lo=std::prev(hi);double w=double(rows-lo->first)/(hi->first-lo->first);t=lo->second*(1-w)+hi->second*w;}
  return t*double(h)*h/(double(nearest_h)*nearest_h);
 }
};
}
